import base64, cv2
from spca_llm_ur5.llm.llmclient import ChatGPTClient

VISION_SYSTEM_PROMPT = """
You are the robot's eyes. You receive a single top-down RGB image of a workbench.
Output a short plain-text description useful for high-level planning.
Do not assume which objects exist; describe only what is visible.
Avoid numeric coordinates; use coarse terms like left/middle/right and near/mid/far.
Mention which colored items are present, which touch the table or each other,
whether the gripper is near/above/touching something, and obvious obstacles.
If uncertain, say "uncertain". Keep it to 4–8 short sentences. Plain text only.
""".strip()

VISION_USER_TEMPLATE = """
Task:
title: {title}
description: {description}

Describe what you see in the provided image, focusing on what matters for this task.
Plain text only.
""".strip()


class SenseLLM:
    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.client = ChatGPTClient(model, output_format=None)

    @staticmethod
    def _bgr_to_b64_jpeg(bgr):
        ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return base64.b64encode(buf).decode("ascii")

    def describe(self, title: str, description: str, bgr_image) -> str:
        if bgr_image is None:
            return ""
        user_text = VISION_USER_TEMPLATE.format(title=title, description=description)
        b64 = self._bgr_to_b64_jpeg(bgr_image)

        messages = [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "input_text",  "text": user_text},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
            ]},
        ]
        return self.client.image_description(messages).strip()
