import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class TreeRegressor:
    
    def trainModel(form):
        df = pd.read_csv('Bank_Personal_Loan_Modelling.csv')

        ran_forest_classifier = RandomForestClassifier(random_state=42)
  
        model_features = df.columns.drop(["ID","PersonalLoan"])
        model_target = "PersonalLoan"


        X_train = df[model_features]
        y_train = df[model_target]

        customer_data = df[model_features].head(1)
    
        customer_data["Age"] = form["age"]
        customer_data["Experience"] = form["experience"]
        customer_data["Income"] = form["income"]
        customer_data["ZIPCode"] = form["ZIP_code"]
        customer_data["Family"] = form["family_size"]
        customer_data["CCAvg"] = form["CCAvg"]
        customer_data["Education"] = form["education"]
        customer_data["Mortgage"] = form["mortgage"]
        customer_data["SecuritiesAccount"] = 0
        customer_data["CDAccount"] = 0
        customer_data["Online"] = 0
        customer_data["CreditCard"] = 0


        # if(form["securities_account"] == "on"):
        #     customer_data["SecuritiesAccount"] = 1
        # else:
        #     customer_data["SecuritiesAccount"] = 0
        # customer_data["CDAccount"] = form["CD_account"]
        # customer_data["Online"] = form["online_bank"]
        # customer_data["CreditCard"] = form["credit_card"]
    


        ran_forest_classifier.fit(X_train, y_train)

        test_predictions = ran_forest_classifier.predict(customer_data)

        if(test_predictions == 0):
            return False
        else:
            return True

    
    

    