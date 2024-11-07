from openai import OpenAI

def query(prompt, model_name, json_format, temperature=0.9 ,timeout=200):

    client = OpenAI(
        base_url='https://tbnx.plus7.plus/v1',
        api_key='sk-rgOjANiJZ0i2xcYrCdD25f94264d4eAbAa377c45Ee17D748'
    )
  
    completion = client.beta.chat.completions.parse(
        model=model_name, 
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=temperature,
        response_format=json_format, 
        timeout=timeout
    )   
    
    program = completion.choices[0].message.parsed   
    return program