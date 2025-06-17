from django.shortcuts import render
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ImageUploadSerializer
from PIL import Image

class ImageToTextView(APIView):
    def post(self, request, format=None):
        serializer = ImageUploadSerializer(data=request.data)
        if serializer.is_valid():
            image = serializer.validated_data['image']

            with open("temp_image.jpg", "wb+") as f:
                for chunk in image.chunks():
                    f.write(chunk)

            try:
                img = Image.open("temp_image.jpg")
                extractedLatex = "test" #run ai model on computer`
            except Exception as e:
                return Response({"error", str(e)}, status=500)

            return Response({"text": extractedLatex}, status=200)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

