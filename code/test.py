from datetime import datetime, timedelta
import uo_pyfetch
# from IPython.display import HTML

# Define bounding box for Newcastle area
bbox = [-1.652756, 54.973377, -1.620483, 54.983721]

# Define bounding box for Newcastle area
variables = uo_pyfetch.get_variables()
print("Available variables:")
for v in variables["Variables"][:100]:
    print(f" - {v['Name']} ({v.get('Units', 'no units')})")

# Fetch available sensor themes
# themes = uo_pyfetch.get_themes()
# print("\nAvailable themes:")
# for t in themes["Themes"]:
#     print(f" - {t['Name']}")


# Fetch available sensor variables
variables = uo_pyfetch.get_variables()
print("Available variables:")
for v in variables["Variables"][:100]:
    print(f" - {v['Name']} ({v.get('Units', 'no units')})")