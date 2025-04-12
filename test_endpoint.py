import requests

# 画像とテキストを送信
url = "http://localhost:8000/predict"
files = {
    'image': open('image.jpg', 'rb'),
}
data = {
    'question': 'What is shown in the image?',
    'options': 'option1,option2,option3,option4'
}
response = requests.post(url, files=files, data=data)
print(response.json())