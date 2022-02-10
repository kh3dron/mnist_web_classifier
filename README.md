# MNIST Web Classifier

Run with: 
NEW_RELIC_CONFIG_FILE=newrelic.ini newrelic-admin run-program uvicorn app.main:app --reload

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
- added new relic one for basic observability
- user database now dumped to local CSV to simplify database accesses and match format of MNIST dataset
- Update model: switch from an MLP to a convolutional network for better performance on translation-blind features
  - was stuck here for a while, translating data shapes & formats between ML libraries
- model re-training is now completely decoupled from webapp. Good design, but currently no way to re-train network. so needs better implementation 


## [todo]
- want to return a percentage distribution of confidences eventually
- dataviz on training convergence would be nice - ROC curve equivalent for non-binary classifier? research
- Deployment:
  - re-containerize & deploy to AWS & add DNS
  - Sagemaker?
  - work queue for training before returning prediction
    - worker pool does training outside the webapp prediction return
- Decouple training from prediction to seperate microservice for better user experience  