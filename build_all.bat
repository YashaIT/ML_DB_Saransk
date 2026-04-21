@echo off
set PYTHONUTF8=1
pyinstaller --noconfirm --onefile --name module_a_agent module_a\agent.py
pyinstaller --noconfirm --onefile --name module_b_agent module_b\agent.py
pyinstaller --noconfirm --onefile --name module_v_agent module_v\agent.py
pyinstaller --noconfirm --onefile --name module_g_agent module_g\agent.py
pyinstaller --noconfirm --onefile --name module_d_agent module_d\agent.py
if not exist Distribution mkdir Distribution
copy /Y dist\module_a_agent.exe Distribution\module_a_agent.exe
copy /Y dist\module_b_agent.exe Distribution\module_b_agent.exe
copy /Y dist\module_v_agent.exe Distribution\module_v_agent.exe
copy /Y dist\module_g_agent.exe Distribution\module_g_agent.exe
copy /Y dist\module_d_agent.exe Distribution\module_d_agent.exe
