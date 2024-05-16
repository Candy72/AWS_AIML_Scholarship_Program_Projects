#Function 1 - SerializeImage

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    s3.download_file(Bucket=bucket, Key=key, Filename='/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "s3_bucket": bucket,
            "s3_key": key,
            "image_data": image_data,
            "inferences": []
        }
    }


#Function 2 - ClassifyImage
import json
import sagemaker
import base64
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2024-05-15-22-45-42-982"

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['body']['image_data'])

    # Instantiate a Predictor
    predictor = Predictor(
        endpoint_name=ENDPOINT,
        serializer = IdentitySerializer(content_type="image/png"),
        deserializer=JSONDeserializer()
    )
    
    # Make a prediction:
    inferences = predictor.predict(image)
    
    print("Predictions:", inferences)

    
    # We return the data back to the Step Function    
    event["body"]["inferences"] = inferences
    return {
        'statusCode': 200,
        'body': json.dumps(event["body"])
    }

# Function 3 - FilterInferenceByConfidence
import json

THRESHOLD = 0.93

def lambda_handler(event, context):
    # Check if 'body' is a string and convert it to dictionary if true
    if isinstance(event['body'], str):
        body = json.loads(event['body'])
    else:
        body = event['body']
    
    # Grab the inferences from the body (corrected from event)
    inferences = body["inferences"]
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(confidence >= THRESHOLD for confidence in inferences)
    
    # Prepare the message based on the threshold check
    if meets_threshold:
        message = "Confidence threshold met. Passing data downstream."
    else:
        message = "CONFIDENCE THRESHOLD NOT MET. Action required."
        raise Exception(message)  # Fails loudly
        
    # Return the result using the correct 'body' dictionary
        return {
            'statusCode': 200,
            'body': json.dumps({
                "s3_bucket": body["s3_bucket"],
                "s3_key": body["s3_key"],
                "inferences": inferences,
                "message": message
            })
        }
