import json
import boto3
import base64

rekognition = boto3.client("rekognition")


def lambda_handler(event, context):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
    }

    if event.get("httpMethod") == "OPTIONS":
        return {"statusCode": 200, "headers": headers, "body": ""}

    try:
        body = json.loads(event.get("body", "{}"))
        image_data = body.get("image", "")

        if not image_data:
            return {
                "statusCode": 400,
                "headers": headers,
                "body": json.dumps({"error": "No image provided"}),
            }

        # Strip base64 prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]

        image_bytes = base64.b64decode(image_data)

        # Detect labels
        label_response = rekognition.detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=10,
            MinConfidence=70,
        )

        # Detect faces
        face_response = rekognition.detect_faces(
            Image={"Bytes": image_bytes},
            Attributes=["ALL"],
        )

        # Safely extract labels
        labels = []
        for label in label_response.get("Labels", []):
            name = label.get("Name") or label.get("name") or "Unknown"
            confidence = round(float(label.get("Confidence") or label.get("confidence") or 0), 1)
            labels.append({"name": name, "confidence": confidence})

        faces = face_response.get("FaceDetails", [])
        top_labels = [l["name"].lower() for l in labels[:5]]
        caption = build_caption(top_labels, faces)

        face_details = []
        for i, face in enumerate(faces):
            age = face.get("AgeRange", {})
            emotions = face.get("Emotions", [])
            emotion = "unknown"
            if emotions:
                top = max(emotions, key=lambda e: e.get("Confidence", 0))
                emotion = top.get("Type", "unknown").lower()
            face_details.append({
                "face": i + 1,
                "age_range": f"{age.get('Low', '?')}-{age.get('High', '?')}",
                "emotion": emotion,
            })

        return {
            "statusCode": 200,
            "headers": headers,
            "body": json.dumps({
                "caption": caption,
                "labels": labels,
                "faces_detected": len(faces),
                "face_details": face_details,
            }),
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            "statusCode": 500,
            "headers": headers,
            "body": json.dumps({"error": str(e)}),
        }


def build_caption(labels, faces):
    if not labels:
        return "Unable to identify content in this image."

    face_count = len(faces)

    if face_count == 1:
        age = faces[0].get("AgeRange", {})
        emotions = faces[0].get("Emotions", [])
        age_str = f"{age.get('Low', '?')}-{age.get('High', '?')} years old" if age else ""
        emotion_str = ""
        if emotions:
            top = max(emotions, key=lambda e: e.get("Confidence", 0))
            if top.get("Confidence", 0) > 70:
                emotion_str = f"appearing {top.get('Type', '').lower()}"
        parts = ["A person"]
        if age_str:
            parts.append(age_str)
        if emotion_str:
            parts.append(emotion_str)
        scene = ", ".join(labels[:2])
        return " ".join(parts) + (f", in a scene with {scene}." if scene else ".")

    elif face_count > 1:
        return f"{face_count} people in a scene with {', '.join(labels[:2])}."

    else:
        if len(labels) == 1:
            return f"An image featuring {labels[0]}."
        elif len(labels) == 2:
            return f"An image featuring {labels[0]} and {labels[1]}."
        else:
            return f"An image of {labels[0]}, with elements of {', '.join(labels[1:3])}."
