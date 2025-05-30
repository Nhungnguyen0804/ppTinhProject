# Cấu hình ứng dụng

# config.py
import os

from flask import current_app
class Config:
    
    
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'secretkey123'
   
