step-by-step guide on how to run the app

the main files contained in the cloud folder

the app.py and model.hdf5 is uplaoded in google cloud, under My First Project -> compute engine -> vm instance -> waste-classifier-project

can run the app.py online by clicking the ssh button, then run with gunicorn using gunicorn --workers 4 --bind 0.0.0.0:5000 app:app

and then run index.html locally, in script.js already specify the port to be the external ip of that vm instance, thus when running index.html locally it will link to the cloud backend to get result, and then result show in frontend.
