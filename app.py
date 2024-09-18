from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()
# Initialize FastAPI app
app = FastAPI()

# Pydantic model for the request body
class ImageRequest(BaseModel):
    image_url: str  # URL of the image containing the PNR

# Initialize OpenAI API
api_key = os.getenv("OPENAI_API_KEY") 
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=api_key)
@app.post("/extract-pnr-from-image")
async def extract_pnr_from_image(image_request: ImageRequest):
    try:
        # OpenAI API call to extract PNR from the image
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the PNR from this image. The PNR is a 10-digit alphanumeric code. Show it in the format **PNR**: **123-4567890**."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_request.image_url,  # Dynamically use the image_url from the request
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        response_dict = response.model_dump()  # Convert to dictionary
        extracted_text = response_dict["choices"][0]["message"]["content"]

        # Assuming the PNR is a 10-character alphanumeric code, extract using regex
        import re
        pnr_match = re.search(r"\*\*(\d{3}-\d{7})\*\*", extracted_text)
        if pnr_match:
            pnr = pnr_match.group(1)
        else:
            raise HTTPException(status_code=400, detail="PNR not found in the image.")

        return {"pnr": pnr}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# To run the FastAPI app using Uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
