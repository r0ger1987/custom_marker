# AWS Bedrock Integration for Marker PDF

This integration adds support for AWS Bedrock foundation models to Marker PDF, allowing you to use Claude, Titan, and other LLMs hosted on AWS Bedrock for enhanced document processing.

## Features

- Full integration with AWS Bedrock foundation models
- Support for Claude 3.5, Claude 3, Titan, and other Bedrock models
- Configuration via environment variables or `.env` file
- Compatible with existing Marker workflows
- Supports both explicit AWS credentials and IAM roles

## Installation

1. Install required dependencies:
```bash
pip install boto3
pip install marker-pdf[full]
```

2. Configure AWS credentials in `.env` file:
```bash
cp .env.example .env
# Edit .env with your AWS credentials and preferences
```

## Configuration

### Environment Variables

Configure Bedrock in your `.env` file:

```bash
# AWS Credentials (optional - uses AWS CLI defaults if not set)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_SESSION_TOKEN=  # Optional for temporary credentials

# AWS Region
AWS_REGION=us-east-1

# Bedrock Model Selection
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0

# Model Parameters
MAX_BEDROCK_TOKENS=8192
TEMPERATURE=0.0
```

### Available Models

- **Claude Models** (Recommended for documents):
  - `anthropic.claude-3-5-sonnet-20241022-v2:0`
  - `anthropic.claude-3-haiku-20240307-v1:0`
  - `anthropic.claude-3-opus-20240229-v1:0`

- **Amazon Titan Models**:
  - `amazon.titan-text-express-v1`
  - `amazon.titan-text-lite-v1`

- **Other Models**:
  - `meta.llama2-70b-chat-v1`
  - `cohere.command-text-v14`
  - `ai21.j2-ultra-v1`

## Usage

### Method 1: Using the Provided Script

```bash
# Convert single PDF
python use_bedrock_marker.py document.pdf --use-llm

# Convert directory of PDFs
python use_bedrock_marker.py /path/to/pdfs/ --use-llm

# Specify output format
python use_bedrock_marker.py document.pdf --format json --output-dir ./results

# Use specific model
python use_bedrock_marker.py document.pdf --model anthropic.claude-3-haiku-20240307-v1:0

# Debug mode with OCR
python use_bedrock_marker.py document.pdf --debug --force-ocr
```

### Method 2: Integration with Existing Code

```python
from marker.converters import PdfConverter
from marker.config import BasePDFConverterConfig
from marker.services.bedrock import BedrockService

# Configure Bedrock service
bedrock_config = {
    "bedrock_model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "aws_region": "us-east-1",
    "aws_access_key_id": "your-key",  # Optional
    "aws_secret_access_key": "your-secret",  # Optional
    "temperature": 0.0,
}

# Initialize service
llm_service = BedrockService(bedrock_config)

# Configure converter
converter_config = BasePDFConverterConfig(
    use_llm=True,
    output_format="markdown"
)

# Create converter with Bedrock
converter = PdfConverter(converter_config)
converter.llm_service = llm_service

# Convert PDF
with open("document.pdf", "rb") as f:
    result = converter(f)

print(result.markdown)
```

### Method 3: Modified CLI Commands

After setting up environment variables:

```bash
# Single file conversion
marker_single document.pdf --use_llm --llm_service bedrock

# Batch conversion
marker /path/to/pdfs --use_llm --llm_service bedrock

# With specific model
BEDROCK_MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0" marker_single document.pdf --use_llm
```

## Testing

Run the test script to verify your configuration:

```bash
python test_bedrock_integration.py
```

This will:
- Check AWS credentials
- Test Bedrock service connection
- Perform sample text and image processing
- Verify model availability

## AWS Permissions

Ensure your AWS credentials have the following IAM permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*:*:provisioned-model/*"
    }
  ]
}
```

## Best Practices

1. **Use IAM Roles**: For production, use IAM roles instead of explicit credentials
2. **Model Selection**: Claude 3.5 Sonnet offers best accuracy for documents
3. **Region Selection**: Choose region closest to your location for lower latency
4. **Cost Management**: Monitor usage as Bedrock charges per token
5. **Error Handling**: Script includes retry logic for rate limits

## Troubleshooting

### Access Denied Error
- Ensure your AWS account has access to the selected Bedrock model
- Some models require explicit access request in AWS Console

### Model Not Found
- Verify model ID is correct and available in your region
- Check AWS Bedrock console for available models

### Rate Limiting
- The integration includes automatic retry with exponential backoff
- Consider using multiple AWS accounts for high-volume processing

### No Credentials Found
- Configure AWS CLI: `aws configure`
- Or set environment variables in `.env`
- Or use IAM roles on EC2/Lambda

## Performance Considerations

- **Latency**: Bedrock adds 1-3 seconds per API call
- **Throughput**: Limited by Bedrock API rate limits
- **Cost**: Varies by model (Claude 3.5 ~$3/million input tokens)
- **Concurrency**: Configure `MAX_CONCURRENCY` in settings

## Support

For issues specific to Bedrock integration:
1. Check AWS Bedrock service health
2. Verify model availability in your region
3. Review CloudWatch logs for detailed errors

For Marker-related issues:
- See main Marker documentation
- Report issues on GitHub