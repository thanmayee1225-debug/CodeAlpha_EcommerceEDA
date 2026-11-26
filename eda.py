print("Hello")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully!")

df = pd.read_csv("Ecommerce_2000.csv")
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
print("\nDuplicate Records:")
print(df[df.duplicated()])
df = df.drop_duplicates()
print("\nDataset Shape After Removing Duplicates:")
print(df.shape)
print("\nDuplicates Remaining:")
print(df.duplicated().sum())
print("\nStatistical Summary:")
print(df.describe())
print("\nProduct Category Counts:")
print(df['ProductCategory'].value_counts())
print("\nPayment Method Counts:")
print(df['PaymentMethod'].value_counts())
print("\nDevice Type Counts:")
print(df['DeviceType'].value_counts())
print("\nMembership Tier Counts:")
print(df['MembershipTier'].value_counts())
print("\nReturned Orders:")
print(df['Returned'].value_counts())
print("\nOrder Season Counts:")
print(df['OrderSeason'].value_counts())
print("\nOrders by Day of Week:")
print(df['OrderDayOfWeek'].value_counts())
df['DayType'] = df['OrderDayOfWeek'].apply(
    lambda day: 'Weekend' if day in ['Saturday', 'Sunday'] else 'Weekday'
)

print("\nWeekday vs Weekend Orders:")
print(df['DayType'].value_counts())
print("\nMinimum Shipping Days:")
print(df['ShippingDays'].min())

print("\nMaximum Shipping Days:")
print(df['ShippingDays'].max())
df['OrderDate'] = pd.to_datetime(df['OrderDate'])

print("\nOrder Date Data Type:")
print(df['OrderDate'].dtype)
df['OrderYear'] = df['OrderDate'].dt.year

print("\nOrders by Year:")
print(df['OrderYear'].value_counts().sort_index())
df['OrderMonth'] = df['OrderDate'].dt.month

print("\nOrders by Month:")
print(df['OrderMonth'].value_counts().sort_index())
print("\nMonthly Sales:")
print(df.groupby('OrderMonth')['FinalAmount'].sum())
monthly_orders = df['OrderMonth'].value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.bar(monthly_orders.index, monthly_orders.values)

plt.title("Monthly Order Distribution")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.show()
monthly_sales = df.groupby('OrderMonth')['FinalAmount'].sum()
# Monthly Sales - Bar Graph

plt.figure(figsize=(10, 5))
plt.bar(monthly_sales.index, monthly_sales.values)

plt.title("Monthly Sales Distribution")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()


# Monthly Sales - Line Graph

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')

# Orders by Payment Method

payment_counts = df['PaymentMethod'].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(payment_counts.index, payment_counts.values)

plt.title("Orders by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
# STEP 1: Meaningful Business Questions

print("\n--- EDA Business Questions ---")
print("1. Which product category generates the highest sales?")
print("2. Which payment method is most commonly used?")
print("3. Which month has the highest sales?")
print("4. Do weekdays have more orders than weekends?")
print("5. Which membership tier has the most customers?")
print("6. What is the return rate?")
print("7. Is there a relationship between discounts and final sales?")

# STEP 2: Sales by Product Category

category_sales = df.groupby('ProductCategory')['FinalAmount'].sum()

print("\nSales by Product Category:")
print(category_sales.sort_values(ascending=False))

# Visualisation

plt.figure(figsize=(10, 5))
plt.bar(category_sales.index, category_sales.values)

plt.title("Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
# STEP 3: Payment Method Analysis

payment_counts = df['PaymentMethod'].value_counts()

print("\nOrders by Payment Method:")
print(payment_counts)

# Visualisation

plt.figure(figsize=(8, 5))
plt.bar(payment_counts.index, payment_counts.values)

plt.title("Orders by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()
# STEP 4: Membership Tier Analysis

membership_counts = df['MembershipTier'].value_counts()

print("\nOrders by Membership Tier:")
print(membership_counts)

# Visualisation

plt.figure(figsize=(8, 5))
plt.bar(membership_counts.index, membership_counts.values)

plt.title("Orders by Membership Tier")
plt.xlabel("Membership Tier")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()

# STEP 5: Return Analysis

return_counts = df['Returned'].value_counts()

print("\nReturn Analysis:")
print(return_counts)

# Calculate Return Percentage

returned_orders = (df['Returned'].astype(str).str.strip().str.lower() == 'yes').sum()

return_percentage = (returned_orders / len(df)) * 100

print("\nTotal Returned Orders:", returned_orders)
print("Return Percentage:", round(return_percentage, 2), "%")

# Visualisation

plt.figure(figsize=(7, 5))

plt.bar(return_counts.index.astype(str), return_counts.values)

plt.title("Returned vs Non-Returned Orders")
plt.xlabel("Return Status")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()
# STEP 6: Discount vs Final Amount

print("\nDiscount vs Final Amount Correlation:")

correlation = df['DiscountPercent'].corr(df['FinalAmount'])

print("Correlation:", round(correlation, 3))

# Visualisation

plt.figure(figsize=(8, 5))

plt.scatter(df['DiscountPercent'], df['FinalAmount'])

plt.title("Discount Percentage vs Final Amount")
plt.xlabel("Discount Percentage")
plt.ylabel("Final Amount")

plt.tight_layout()
plt.show()

# STEP 7: Shipping Days Analysis

print("\nShipping Days Analysis:")

print("Average Shipping Days:", round(df['ShippingDays'].mean(), 2))
print("Minimum Shipping Days:", df['ShippingDays'].min())
print("Maximum Shipping Days:", df['ShippingDays'].max())

# Visualisation - Histogram

plt.figure(figsize=(8, 5))

plt.hist(df['ShippingDays'], bins=8, edgecolor='black')

plt.title("Distribution of Shipping Days")
plt.xlabel("Shipping Days")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()
# STEP 8: Outlier Detection

# Final Amount
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['FinalAmount'])
plt.title("Outliers in Final Amount")
plt.show()

# Unit Price
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['UnitPrice'])
plt.title("Outliers in Unit Price")
plt.show()

# Quantity
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['Quantity'])
plt.title("Outliers in Quantity")
plt.show()

# Shipping Days
plt.figure(figsize=(8, 4))
sns.boxplot(x=df['ShippingDays'])
plt.title("Outliers in Shipping Days")
plt.show()
# STEP 9: Correlation Analysis

numeric_df = df.select_dtypes(include='number')

correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualisation - Heatmap

plt.figure(figsize=(12, 8))

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()
# STEP 10: Data Quality Validation

print("\n--- Data Quality Checks ---")

# 1. Negative Unit Prices
print("Negative Unit Prices:", (df['UnitPrice'] < 0).sum())

# 2. Zero or Negative Quantity
print("Invalid Quantities:", (df['Quantity'] <= 0).sum())

# 3. Negative Final Amounts
print("Negative Final Amounts:", (df['FinalAmount'] < 0).sum())

# 4. Invalid Discount Percentages
print("Invalid Discounts:",
      ((df['DiscountPercent'] < 0) |
       (df['DiscountPercent'] > 100)).sum())

# 5. Invalid Review Scores
print("Invalid Review Scores:",
      ((df['ReviewScore'] < 1) |
       (df['ReviewScore'] > 5)).sum())

# 6. Invalid Shipping Days
print("Invalid Shipping Days:", (df['ShippingDays'] <= 0).sum())
# STEP 11: Hypothesis Testing

print("\n--- Hypothesis Testing ---")

# Compare low-discount and high-discount orders
low_discount = df[df['DiscountPercent'] <= 10]
high_discount = df[df['DiscountPercent'] > 10]

print("Average Final Amount (Low Discount):",
      low_discount['FinalAmount'].mean())

print("Average Final Amount (High Discount):",
      high_discount['FinalAmount'].mean())
# STEP 12: Final EDA Insights

print("\n========== FINAL EDA INSIGHTS ==========")

# 1. Highest-selling product category
category_sales = df.groupby('ProductCategory')['FinalAmount'].sum()
print("Highest Sales Category:", category_sales.idxmax())

# 2. Most popular payment method
payment_counts = df['PaymentMethod'].value_counts()
print("Most Used Payment Method:", payment_counts.idxmax())

# 3. Highest-sales month
monthly_sales = df.groupby('OrderMonth')['FinalAmount'].sum()
print("Highest Sales Month:", monthly_sales.idxmax())

# 4. Weekday vs Weekend orders
day_counts = df['DayType'].value_counts()
print("Orders by Day Type:")
print(day_counts)

# 5. Return percentage
return_count = (df['Returned'].str.lower() == 'yes').sum()
return_percentage = (return_count / len(df)) * 100
print("Return Percentage:", round(return_percentage, 2), "%")

# 6. Average order value
average_order_value = df['FinalAmount'].mean()
print("Average Order Value:", round(average_order_value, 2))

# 7. Total revenue
total_revenue = df['FinalAmount'].sum()
print("Total Revenue:", round(total_revenue, 2))

print("======================================")