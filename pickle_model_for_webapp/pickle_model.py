import pickle
# MUST TO TRAIN THE MODEL IN SEPARATE FILE AND IMPORT IT
from pickle_model_for_webapp.train_pickle_pipeline_for_webapp import train_model


# ======================================================================================================================
# THIS SCRIPT IS MANDATORY IN ORDER TO PRODUCE A PICKLE MODEL THAT RUNS IS HEROKU. THE PICKLE MUST BE SAVED INSIDE THE
# MODULE NAMED "__main__", SO THAT WHEN UN-PICKLING THE MODEL THE dummy FUNCTION CAN BE LOCATED AND READ. ADDITIONALLY,
# THE FILE THAT CONTAINS THE MODEL THAT WILL BE TRAINED IN THE WHOLE DATASET (train_pickle_pipeline_for_webapp) MUST BE
# IMPORTED AS WELL IN ORDER FOR THIS TO WORK (it cannot just be inserted in the "__main__" module, it MUST be in
# separate file that MUST be imported)

# !!! CAUTION !!!

# IF NOT DONE THIS WAY, WHEN LOADING THE MODEL IN 'app.py' IT WILL NOT LOCATE THE
# 'dummy' FUNCTION THAT IS USED FOR THE TF-IDF PRE-PROCESSOR AND TOKENIZER. IF THE MODEL IS INSERTED INSIDE THE MODULE
# __name__ == "__main__", THEN IT WILL RECOGNIZE THE 'dummy' FUNCTION, BUT THEN THE PICKLE LOADED MODEL WILL NOT ABLE TO
# BE USED FROM WEB APP ROUTES
# ======================================================================================================================

if __name__ == "__main__":
    model = train_model()  # MUST TO TRAIN THE MODEL IN SEPARATE FILE

    # Saving model to disk
    pickle.dump(model, open('model.pkl', 'wb'))  # MUST TO SAVE THE MODEL UNDER MODULE "__main__"
