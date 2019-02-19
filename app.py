from inat_vision_api import InatVisionAPI

api = InatVisionAPI( )

if __name__ == "__main__":
    api.app.run( host="0.0.0.0", port=6006 )
