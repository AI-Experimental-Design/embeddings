
#!/usr/bin/env python3
"""Calculate variance of numbers from stdin."""

import sys
import numpy as np

values = [float(line) for line in sys.stdin if line.strip()]
print(np.var(values))

