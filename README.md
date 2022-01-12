# MNIST Web Classifier

Run with: 
uvicorn app.main:app --reload

### Changelog 
## [1.0]
- generated and trained model off of mnist dataset
- put model behind API on webapp with Flask
- javascript function for user to drwa their own digit and have it classified
- application containerized and deployed to AWS
## [1.1] (Revitalized)
- app moved from flask to fastAPI
- Jinja and CSS used to make UI 40% less dogshit
## [1.2]
- DB and API to store past performance of predictions
- added form to label your own drawing
- Db that stores performance of model
- CRUDs, APIs and schemas for history DB
- JS function that draws MNIST formatted digits
- Downloaded original CSV dataset for use in "explore data" page, view some sample data
  - cut data down to first 1000 rows from MNSIT for faster loading
## [1.3]
- imported model generation program from kaggle
- model generation produces modelstats.txt metadata json, which is shown on site
- the data pipeline is [up]
  - each time a user uses the model, a new data object is created: user drawing and user label. this is added to a DB of user created data
  - model trainer launches, opens both CSV MNIST dataset and the user dataset, trains
  - re-generates metadata page, re-deploys model to classify page
## [1.4]
- generate requirements.txt with pipreqs
- build docker with docker build -t imagename
- uplodaed to ECR
## [1.5]
- Service-by-service deployment to AWS
- Sagemaker


## [todo]
- dataviz on training convergence would be nice - ROC curve equivalent for non-binary classifier? research
- Deployment:
  - re-containerize & deploy to AWS
  - add DNS