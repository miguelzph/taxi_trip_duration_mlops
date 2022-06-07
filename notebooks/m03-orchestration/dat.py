from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

date="2021-03-15"

s = datetime.strptime(date, "%Y-%m-%d")
print(str(((s - timedelta(days=60)).month)).zfill(2))

print(s + relativedelta(months=-2))


