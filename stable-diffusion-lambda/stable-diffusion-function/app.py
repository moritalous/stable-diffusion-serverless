import base64
import json
import os

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained('/model')
pipe = pipe.to('cpu')

def lambda_handler(event, context):

    width = int(os.environ['WIDTH'])
    height = int(os.environ['HEIGHT'])

    try:
        os.remove('/tmp/image.png')
    except:
        pass

    body = json.loads(event['body'])
    prompt = body['prompt']
    print(prompt)

    image = pipe(prompt, width=width, height=height, guidance_scale=7.5).images[0]
    image.save('/tmp/image.png')

    with open('/tmp/image.png', 'rb') as f:
        base64_img = base64.b64encode(f.read()).decode('utf-8')

    return {
        "statusCode": 200,
        "headers": { "Content-Type": "image/png" },
        "body": base64_img,
        "isBase64Encoded": True
    }
