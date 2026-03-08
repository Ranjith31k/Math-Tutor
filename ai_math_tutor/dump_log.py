import sys
sys.stdout = open('clean_log.txt', 'w', encoding='utf-8')
sys.stderr = sys.stdout
import run_smoke_test
run_smoke_test.run()
