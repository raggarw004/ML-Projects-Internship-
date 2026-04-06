import pandas as pd

def load_data():
    return pd.DataFrame({
        "product_id": ["p1", "p2", "p3", "p4", "p5"],
        "product_name": [
            "Running Shoes",
            "Casual Sneakers",
            "Gaming Laptop",
            "Business Laptop",
            "Sports Sandals"
        ],
        "description": [
            "lightweight sporty running shoes for exercise and fitness",
            "comfortable casual sneakers for daily wear and walking",
            "high performance gaming laptop with strong graphics",
            "professional business laptop for office productivity and work",
            "open sporty sandals for outdoor summer activity"
        ]
    })