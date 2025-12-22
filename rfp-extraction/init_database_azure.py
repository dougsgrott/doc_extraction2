

# Run the initialization
import os
import sys
sys.path.insert(0, 'rfp-extraction')
from shared.models.db import init_database
engine = init_database(os.environ['DATABASE_URL'])
print('✅ Tables created in Azure database!')