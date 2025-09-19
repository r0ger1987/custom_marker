import json
import time
from typing import List, Annotated, T, Optional

import PIL
from PIL import Image
import boto3
from botocore.exceptions import ClientError
from marker.logger import get_logger
from pydantic import BaseModel

from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class BedrockService(BaseService):
    bedrock_model_id: Annotated[
        str, "The model ID to use for AWS Bedrock."
    ] = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    aws_region: Annotated[
        str, "The AWS region for Bedrock."
    ] = "us-east-1"
    aws_access_key_id: Annotated[
        Optional[str], "AWS Access Key ID for Bedrock."
    ] = None
    aws_secret_access_key: Annotated[
        Optional[str], "AWS Secret Access Key for Bedrock."
    ] = None
    aws_session_token: Annotated[
        Optional[str], "AWS Session Token for temporary credentials."
    ] = None
    max_bedrock_tokens: Annotated[
        int, "The maximum number of tokens to use for a single Bedrock request."
    ] = 8192
    temperature: Annotated[
        float, "The temperature for text generation."
    ] = 0.0

    def process_images(self, images: List[Image.Image]) -> List[dict]:
        """Process images for Bedrock Claude models."""
        processed_images = []
        for img in images:
            processed_images.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/webp",
                    "data": self.img_to_base64(img),
                }
            })
        return processed_images

    def validate_response(self, response_text: str, schema: type[T]) -> T:
        """Validate and parse response text according to schema."""
        response_text = response_text.strip()
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        try:
            # Try to parse as JSON first
            out_schema = schema.model_validate_json(response_text)
            out_json = out_schema.model_dump()
            return out_json
        except Exception:
            try:
                # Re-parse with fixed escapes
                escaped_str = response_text.replace("\\", "\\\\")
                out_schema = schema.model_validate_json(escaped_str)
                return out_schema.model_dump()
            except Exception as e:
                logger.error(f"Failed to parse response: {e}")
                return {}

    def get_client(self):
        """Get AWS Bedrock runtime client."""
        session_params = {
            "region_name": self.aws_region
        }

        # Use explicit credentials if provided, otherwise use default AWS credential chain
        if self.aws_access_key_id and self.aws_secret_access_key:
            session_params["aws_access_key_id"] = self.aws_access_key_id
            session_params["aws_secret_access_key"] = self.aws_secret_access_key

            if self.aws_session_token:
                session_params["aws_session_token"] = self.aws_session_token

        session = boto3.Session(**session_params)
        return session.client('bedrock-runtime')

    def _prepare_claude_request(self, prompt: str, image_data: list, schema: type[BaseModel]) -> dict:
        """Prepare request body for Claude models on Bedrock."""
        schema_example = schema.model_json_schema()

        system_prompt = f"""
Follow the instructions given by the user prompt. You must provide your response in JSON format matching this schema:

{json.dumps(schema_example, indent=2)}

Respond only with the JSON schema, nothing else. Do not include ```json, ```, or any other formatting.
""".strip()

        messages = [
            {
                "role": "user",
                "content": [
                    *image_data,
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_bedrock_tokens,
            "system": system_prompt,
            "messages": messages,
            "temperature": self.temperature
        }

    def _prepare_titan_request(self, prompt: str, image_data: list, schema: type[BaseModel]) -> dict:
        """Prepare request body for Titan models on Bedrock."""
        schema_example = schema.model_json_schema()

        full_prompt = f"""
Follow the instructions below and provide your response in JSON format matching this schema:

{json.dumps(schema_example, indent=2)}

Instructions: {prompt}

Respond only with the JSON, nothing else.
""".strip()

        # Titan models have different image format
        titan_images = []
        if image_data:
            for img_dict in image_data:
                if img_dict.get("type") == "image":
                    titan_images.append({
                        "image": img_dict["source"]["data"]
                    })

        request_body = {
            "inputText": full_prompt,
            "textGenerationConfig": {
                "maxTokenCount": min(self.max_bedrock_tokens, 4096),  # Titan max is 4096
                "temperature": self.temperature,
                "topP": 0.9
            }
        }

        if titan_images:
            request_body["images"] = titan_images

        return request_body

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        # Prepare request based on model type
        if "claude" in self.bedrock_model_id.lower():
            request_body = self._prepare_claude_request(prompt, image_data, response_schema)
        elif "titan" in self.bedrock_model_id.lower():
            request_body = self._prepare_titan_request(prompt, image_data, response_schema)
        else:
            # Default to Claude format for other models
            request_body = self._prepare_claude_request(prompt, image_data, response_schema)

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                # Configure client timeout
                config = client._client_config
                config.read_timeout = timeout

                response = client.invoke_model(
                    modelId=self.bedrock_model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json"
                )

                # Parse response
                response_body = json.loads(response['body'].read())

                # Extract content based on model type
                if "claude" in self.bedrock_model_id.lower():
                    response_text = response_body.get('content', [{}])[0].get('text', '')
                elif "titan" in self.bedrock_model_id.lower():
                    results = response_body.get('results', [{}])
                    response_text = results[0].get('outputText', '') if results else ''
                else:
                    # Try common response formats
                    response_text = (
                        response_body.get('completion', '') or
                        response_body.get('content', [{}])[0].get('text', '') or
                        response_body.get('text', '')
                    )

                if response_text:
                    return self.validate_response(response_text, response_schema)
                else:
                    logger.error(f"Empty response from Bedrock model")

            except ClientError as e:
                error_code = e.response['Error']['Code']
                error_message = e.response['Error']['Message']

                if error_code in ['ThrottlingException', 'TooManyRequestsException']:
                    if tries == total_tries:
                        logger.error(
                            f"Rate limit error: {error_message}. Max retries reached. (Attempt {tries}/{total_tries})"
                        )
                        break
                    else:
                        wait_time = tries * self.retry_wait_time
                        logger.warning(
                            f"Rate limit error: {error_message}. Retrying in {wait_time} seconds... (Attempt {tries}/{total_tries})"
                        )
                        time.sleep(wait_time)
                elif error_code == 'AccessDeniedException':
                    logger.error(f"Access denied to Bedrock model: {error_message}")
                    break
                elif error_code == 'ResourceNotFoundException':
                    logger.error(f"Model not found: {self.bedrock_model_id}. Error: {error_message}")
                    break
                else:
                    logger.error(f"AWS Bedrock error: {error_code} - {error_message}")
                    if tries == total_tries:
                        break
                    time.sleep(self.retry_wait_time)

            except Exception as e:
                logger.error(f"Error during Bedrock API call: {e}")
                if tries == total_tries:
                    break
                time.sleep(self.retry_wait_time)

        return {}