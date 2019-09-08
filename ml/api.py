# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 02:35:47 2019

@author: Devdarshan
"""
import synd
from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class predict(Resource):
    def get(self, query):
        return {'data': manual_predict(query)}

api.add_resource(predict, '/synd/<query>')

if __name__ == '__main__':
     app.run()