import base64
import json
import os

from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained('/model')
pipe = pipe.to('cpu')


def lambda_handler(event, context):

    width = int(os.getenv('WIDTH', '512'))
    height = int(os.getenv('HEIGHT', '512'))
    num_inference_steps = int(os.getenv('NUM_INFERENCE_STEPS', '50'))
    guidance_scale = float(os.getenv('GUIDANCE_SCALE', '7.5'))
    eta = float(os.getenv('ETA', '0.0'))

    try:
        os.remove('/tmp/image.png')
    except:
        pass

    body = json.loads(event['body'])
    prompt = body['prompt']
    print(prompt)

    image = pipe(prompt,
                 width=width,
                 height=height,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale,
                 eta=eta
                 ).images[0]

    image.save('/tmp/image.png')

    with open('/tmp/image.png', 'rb') as f:
        base64_img = base64.b64encode(f.read()).decode('utf-8')

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "image/png"},
        "body": base64_img,
        "isBase64Encoded": True
    }
