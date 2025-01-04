from zytron_models.models import AzureOpenAI

# Initialize Azure OpenAI
model = AzureOpenAI()

# Run the model
model(
    "Create a youtube script for a video on how to use the zytron"
    " framework"
)
