import logging
import os

from slack_bolt import App
from slack_bolt.adapter.aws_lambda import SlackRequestHandler

# process_before_response must be True when running on FaaS
app = App(process_before_response=True, logger=logging.Logger(name='app'))


@app.middleware  # or app.use(log_request)
def log_request(logger, body, next):
    logger.debug(body)
    return next()


command = "/stablediffusion"
command2 = "/translate_stablediffusion"


def respond_to_slack_within_3_seconds(body, ack):
    if body.get("text") is None:
        ack(f":x: Usage: {command} (description here)")
    else:
        title = body["text"]
        ack(f"Accepted! (task: {title})")

def respond_to_slack_within_3_seconds2(body, ack):
    if body.get("text") is None:
        ack(f":x: Usage: {command2} (description here)")
    else:
        title = body["text"]
        ack(f"Accepted! (task: {title})")


def process_request(respond, body):

    prompt = body["text"]
    channel_id = body['channel_id']

    result = app.client.files_upload(
        channels=channel_id,
        initial_comment=prompt,
        file=generate_image(prompt),
    )
    print(result)

    respond(f"Completed! (task: {prompt})")

def process_request2(respond, body):

    prompt = body["text"]
    channel_id = body['channel_id']

    import boto3
    translate = boto3.client(service_name='translate', region_name='ap-northeast-1', use_ssl=True)
    result = translate.translate_text(Text=prompt, 
                SourceLanguageCode="ja", TargetLanguageCode="en")
    
    prompt = result.get('TranslatedText')

    result = app.client.files_upload(
        channels=channel_id,
        initial_comment=prompt,
        file=generate_image(prompt),
    )
    print(result)

    respond(f"Completed! (task: {prompt})")

app.command(command)(ack=respond_to_slack_within_3_seconds,
                     lazy=[process_request])

app.command(command2)(ack=respond_to_slack_within_3_seconds2,
                     lazy=[process_request2])

SlackRequestHandler.clear_all_log_handlers()
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.DEBUG)


def generate_image(prompt: str):

    from diffusers import StableDiffusionPipeline

    width = int(os.getenv('WIDTH', '512'))
    height = int(os.getenv('HEIGHT', '512'))
    num_inference_steps = int(os.getenv('NUM_INFERENCE_STEPS', '50'))
    guidance_scale = float(os.getenv('GUIDANCE_SCALE', '7.5'))
    eta = float(os.getenv('ETA', '0.0'))

    try:
        os.remove('/tmp/image.png')
    except:
        pass

    pipe = StableDiffusionPipeline.from_pretrained('/model')
    pipe = pipe.to('cpu')

    print(prompt)

    image = pipe(prompt,
                 width=width,
                 height=height,
                 num_inference_steps=num_inference_steps,
                 guidance_scale=guidance_scale,
                 eta=eta
                 ).images[0]

    image.save('/tmp/image.png')

    return '/tmp/image.png'


def lambda_handler(event, context):
    slack_handler = SlackRequestHandler(app=app)
    return slack_handler.handle(event, context)
