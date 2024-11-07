from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # For flashing messages

# Route for the main page
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    if request.method == "POST":
        # Check if the file was uploaded
        file = request.files.get("file")
        if not file:
            flash("Please upload a CSV file.", "danger")
            return redirect(url_for("index"))
        
        # Load the data
        try:
            sales_data = pd.read_csv(file)
            sales_data['InvoiceDate'] = pd.to_datetime(sales_data['InvoiceDate'], dayfirst=True)

            # Preprocess the data
            filtered_data = sales_data[sales_data['Country'] == 'United Kingdom']
            basket = filtered_data.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
            basket = basket.applymap(lambda x: x > 0)
            basket = basket.loc[:, basket.sum() > 5]

            # Generate frequent itemsets
            frequent_itemsets = apriori(basket, min_support=0.02, use_colnames=True)
            if frequent_itemsets.empty:
                flash("No frequent itemsets found. Try adjusting the dataset.", "warning")
                return redirect(url_for("index"))

            # Generate association rules
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)
            if rules.empty:
                flash("No association rules generated. Adjust support or confidence.", "warning")
                return redirect(url_for("index"))

            # Sort rules and select top recommendations
            top_rules = rules.sort_values('confidence', ascending=False).head(5)
            recommendations = [
                {
                    "rule": f"{list(row['antecedents'])} -> {list(row['consequents'])}",
                    "confidence": round(row['confidence'], 2)
                } for _, row in top_rules.iterrows()
            ]

        except Exception as e:
            flash(f"Error processing file: {str(e)}", "danger")
            return redirect(url_for("index"))

    return render_template("product_recomendation.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
