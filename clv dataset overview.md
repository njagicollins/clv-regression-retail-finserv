# Customer Lifetime Value Dataset

This dataset contains customer-level records from a retail/financial services context, used to predict the total economic value a customer is expected to generate over their relationship with the business. The target variable is driven primarily by spending behaviour, tenure, and customer engagement patterns.

---

## Columns

| Column | Description |
|---|---|
| `customer_id` | Unique customer identifier |
| `age` | Customer age in years |
| `gender` | Male / Female |
| `region` | Geographic region of the customer |
| `customer_segment` | Premium / Standard / Basic |
| `tenure_months` | Number of months the customer has been active |
| `avg_monthly_spend` | Average amount spent per month (USD) |
| `avg_transaction_value` | Average value per individual transaction (USD) |
| `purchase_frequency` | Number of purchases made per year |
| `num_products_owned` | Number of distinct products or services held |
| `engagement_score` | Composite engagement score (0–100) |
| `churn_risk_score` | Estimated likelihood of churn (0–100, higher = riskier) |
| `satisfaction_score` | Customer satisfaction rating (1–5) |
| `email_open_rate` | Percentage of marketing emails opened (0–100) |
| `days_since_last_purchase` | Number of days since the most recent transaction |
| `num_support_calls` | Total number of support interactions recorded |
| `discount_usage_rate` | Proportion of purchases made using a discount (0–100) |
| `preferred_channel` | Primary purchase channel (Online / In-Store / Mobile / Phone) |
| `payment_method` | Preferred payment method |
| `customer_lifetime_value` | **Target** — estimated total value of the customer (USD, continuous, right-skewed) |
