#!/usr/bin/env python
# coding: utf-8

# In[6]:


#import generated japanese scores into data 
import pandas as pd
import numpy as np
data=pd.read_csv('scorejapanese.csv',index_col=0)


# In[8]:


import numpy as np
#pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False) 
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.max_rows', None) 


# In[10]:


import openai
import os
import IPython
from dotenv import load_dotenv
load_dotenv()

get_ipython().run_line_magic('env', 'OPENAI_API_KEY=your openai key')
openai.api_key = os.getenv('OPENAI_API_KEY')


# In[13]:


#first step zero-shot prompting
import json
CONTENT = """
Please use one sentence each item to depict two kinds of credit scores: demographical and financial behavioral scores provided in the datasets below. Financial behavioral scores are more important.
demo_score   beh_score                   
0.251954     0.153432
0.251703     0.152459
0.146897     0.154106
0.249124     0.153890
0.249311     0.151596
0.249013     0.154339
0.244952     0.154192
0.249356     0.153884
0.248947     0.151403
0.178513     0.154106
0.251954     0.163728
0.175308     0.163607
0.251055     0.163793
0.249691     0.163504
0.249033     0.163432
0.249032     0.163762
0.249326     0.163991
0.249157     0.163858
0.249104     0.163582
0.247162     0.164278
0.248738     0.151819
0.249224     0.152031
0.147419     0.152025
0.251059     0.152215
0.237427     0.152451
0.251315     0.151456
0.249255     0.151789
0.249392     0.151750
0.103024     0.151602
0.249518     0.152581
0.250799     0.151878
0.251829     0.150026
0.247590     0.152408
0.160311     0.150166
0.249675     0.151289
0.249419     0.149007
0.249052     0.149896
0.249328     0.150244
0.155083     0.150071
0.248308     0.150259
0.248676     0.148899
0.249349     0.160536
0.249409     0.160293
0.170725     0.160399
0.247699     0.160252
0.250551     0.160297
0.249350     0.160822
0.249770     0.160217
0.249341     0.160445
0.249759     0.160376
0.249339     0.160235
0.249383     0.160198
0.251575     0.160218
0.250551     0.160320
0.249759     0.160285
0.173485     0.160553
0.249033     0.160205
0.249313     0.160609
0.249136     0.160229
0.167255     0.160285
0.248738     0.160668
0.242375     0.161715
0.250210     0.154436
0.249236     0.158589
0.249100     0.156592
0.249033     0.156733
0.249034     0.156505
0.249667     0.150227
0.248947     0.155383
0.249350     0.156277
0.243928     0.156493
0.214716     0.156473
0.249033     0.156626
0.249202     0.136297
0.249919     0.156447
0.248975     0.156611
0.242375     0.156727
0.247590     0.155087
0.246845     0.156672
0.249104     0.155597
0.249035     0.156693
0.153814     0.140189
0.249495     0.150024
0.249667     0.160546
0.172147     0.151851
0.249246     0.156883
0.251315     0.152069
0.169257     0.152412
0.249349     0.152057
0.247355     0.152057
0.249349     0.151130
0.249255     0.154393
0.250799     0.157031
0.248947     0.156738
0.249759     0.156643
0.249582     0.160184
0.249202     0.160265
0.245139     0.160691
0.170886     0.160210
0.249423     0.160512
0.251703     0.160242
0.249013     0.160180
0.249255     0.160454
0.249366     0.160232
0.249577     0.160232
0.249013     0.163797
0.249013     0.163796
0.249093     0.163793
0.249093     0.163794
0.249349     0.163793
0.251583     0.163797
0.251583     0.163796
0.250891     0.163793
0.251445     0.163794
0.175177     0.163793
0.249124     0.163369
0.249033     0.162328
0.250433     0.162980
0.249350     0.162837
0.250419     0.162747
0.248947     0.163265
0.247162     0.162416
0.249268     0.162925
0.249255     0.163039
0.249349     0.162701
"""
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": CONTENT},
    ],
    temperature=0,
stream=True)

#json.loads(str(response))
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")


# In[14]:


#second step deepseek zero-shot prompting (using llama environments). For data containing more sequences, patch them every 100.
from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "your Nvidia API Key"
)

completion = client.chat.completions.create(
  model="deepseek-ai/deepseek-r1",
  messages=[{"role":"user","content":"Please output the final classification results given demographical and financial behavioral scores provided in the datasets below. Output: index no, 0 or 1 (0 denotes normal and 1 denotes risky). Note that financial behavioral scores are more important. Data: 1. The demographic score is 0.366717, while the more crucial financial behavioral score is 0.271049.|2. With a demographic score of 0.366435, the financial behavioral score remains significant at 0.271037.|3. A demographic score of 0.233987 contrasts with a more important financial behavioral score of 0.271065.|4. The demographic score is 0.363350, but the financial behavioral score of 0.271061 takes precedence.|5. Despite a demographic score of 0.363630, the financial behavioral score of 0.271146 is more critical.|6. The demographic score is 0.363287, yet the financial behavioral score of 0.271085 is prioritized.|7. A demographic score of 0.359488 is overshadowed by the financial behavioral score of 0.271052.|8. The demographic score is 0.363737, but the financial behavioral score of 0.271050 is more significant.|9. With a demographic score of 0.363436, the financial behavioral score of 0.271101 is more important.|10. The demographic score is 0.261735, while the financial behavioral score of 0.271065 holds more weight.|11. A demographic score of 0.366717 is less important than the financial behavioral score of 0.268132.|12. The demographic score is 0.259263, but the financial behavioral score of 0.268242 is prioritized.|13. Despite a demographic score of 0.365707, the financial behavioral score of 0.267849 is more critical.|14. The demographic score is 0.364135, yet the financial behavioral score of 0.268118 takes precedence.|15. A demographic score of 0.363159 is overshadowed by the financial behavioral score of 0.268242.|16. The demographic score is 0.363312, but the financial behavioral score of 0.268041 is more significant.|17. With a demographic score of 0.363667, the financial behavioral score of 0.267546 is more important.|18. The demographic score is 0.363477, while the financial behavioral score of 0.267859 holds more weight.|19. A demographic score of 0.363575 is less important than the financial behavioral score of 0.268013.|20. The demographic score is 0.361667, but the financial behavioral score of 0.268304 is prioritized.|21. Despite a demographic score of 0.363242, the financial behavioral score of 0.271049 is more critical.|22. The demographic score is 0.363568, yet the financial behavioral score of 0.271092 takes precedence.|23. A demographic score of 0.234048 is overshadowed by the financial behavioral score of 0.271104.|24. The demographic score is 0.365762, but the financial behavioral score of 0.271048 is more significant.|25. With a demographic score of 0.351403, the financial behavioral score of 0.271095 is more important.|26. The demographic score is 0.366000, while the financial behavioral score of 0.271083 holds more weight.|27. A demographic score of 0.363697 is less important than the financial behavioral score of 0.271049.|28. The demographic score is 0.363753, but the financial behavioral score of 0.270974 is prioritized.|29. Despite a demographic score of 0.200416, the financial behavioral score of 0.271111 is more critical.|30. The demographic score is 0.363853, yet the financial behavioral score of 0.271116 takes precedence.|31. A demographic score of 0.365417 is overshadowed by the financial behavioral score of 0.271062.|32. The demographic score is 0.366577, but the financial behavioral score of 0.270795 is more significant.|33. With a demographic score of 0.361812, the financial behavioral score of 0.271056 is more important.|34. The demographic score is 0.243248, while the financial behavioral score of 0.270807 holds more weight.|35. A demographic score of 0.364097 is less important than the financial behavioral score of 0.270945.|36. The demographic score is 0.363766, but the financial behavioral score of 0.270717 is prioritized.|37. Despite a demographic score of 0.363338, the financial behavioral score of 0.270787 is more critical.|38. The demographic score is 0.363742, yet the financial behavioral score of 0.270831 takes precedence.|39. A demographic score of 0.238119 is overshadowed by the financial behavioral score of 0.270798.|40. The demographic score is 0.362825, but the financial behavioral score of 0.270815 is more significant.|41. With a demographic score of 0.363327, the financial behavioral score of 0.270633 is more important.|42. The demographic score is 0.363711, while the financial behavioral score of 0.270067 holds more weight.|43. A demographic score of 0.363924 is less important than the financial behavioral score of 0.270092.|44. The demographic score is 0.253257, but the financial behavioral score of 0.270027 is prioritized.|45. Despite a demographic score of 0.362215, the financial behavioral score of 0.270148 is more critical.|46. The demographic score is 0.365136, yet the financial behavioral score of 0.270097 takes precedence.|47. A demographic score of 0.363674 is overshadowed by the financial behavioral score of 0.269857.|48. The demographic score is 0.363908, but the financial behavioral score of 0.270133 is more significant.|49. With a demographic score of 0.363746, the financial behavioral score of 0.270075 is more important.|50. The demographic score is 0.364217, while the financial behavioral score of 0.270112 holds more weight.|51. A demographic score of 0.363689 is less important than the financial behavioral score of 0.270126.|52. The demographic score is 0.363677, but the financial behavioral score of 0.270141 is prioritized.|53. Despite a demographic score of 0.366291, the financial behavioral score of 0.270132 is more critical.|54. The demographic score is 0.365136, yet the financial behavioral score of 0.270114 takes precedence.|55. A demographic score of 0.364217 is overshadowed by the financial behavioral score of 0.270157.|56. The demographic score is 0.256732, but the financial behavioral score of 0.270079 is more significant.|57. With a demographic score of 0.363153, the financial behavioral score of 0.270155 is more important.|58. The demographic score is 0.363646, while the financial behavioral score of 0.270050 holds more weight.|59. A demographic score of 0.363448 is less important than the financial behavioral score of 0.270127.|60. The demographic score is 0.249213, but the financial behavioral score of 0.270052 is prioritized.|61. Despite a demographic score of 0.363242, the financial behavioral score of 0.270233 is more critical.|62. The demographic score is 0.356623, yet the financial behavioral score of 0.270847 takes precedence.|63. A demographic score of 0.364745 is overshadowed by the financial behavioral score of 0.270504.|64. The demographic score is 0.363516, but the financial behavioral score of 0.270997 is more significant.|65. With a demographic score of 0.363312, the financial behavioral score of 0.270856 is more important.|66. The demographic score is 0.363159, while the financial behavioral score of 0.270824 holds more weight.|67. A demographic score of 0.363135 is less important than the financial behavioral score of 0.270923.|68. The demographic score is 0.363763, but the financial behavioral score of 0.269253 is prioritized.|69. Despite a demographic score of 0.363436, the financial behavioral score of 0.270601 is more critical.|70. The demographic score is 0.363913, yet the financial behavioral score of 0.270928 takes precedence.|71. A demographic score of 0.358278 is overshadowed by the financial behavioral score of 0.270928.|72. The demographic score is 0.326617, but the financial behavioral score of 0.270911 is more significant.|73. With a demographic score of 0.363148, the financial behavioral score of 0.270867 is more important.|74. The demographic score is 0.363537, while the financial behavioral score of 0.264370 holds more weight.|75. A demographic score of 0.364407 is less important than the financial behavioral score of 0.270870.|76. The demographic score is 0.362931, but the financial behavioral score of 0.270871 is prioritized.|77. Despite a demographic score of 0.356623, the financial behavioral score of 0.270883 is more critical.|78. The demographic score is 0.361812, yet the financial behavioral score of 0.270591 takes precedence.|79. A demographic score of 0.361340 is overshadowed by the financial behavioral score of 0.270856.|80. The demographic score is 0.363575, but the financial behavioral score of 0.270670 is more significant.|81. With a demographic score of 0.363175, the financial behavioral score of 0.270878 is more important.|82. The demographic score is 0.237415, while the financial behavioral score of 0.265816 holds more weight.|83. A demographic score of 0.363867 is less important than the financial behavioral score of 0.268918.|84. The demographic score is 0.363763, but the financial behavioral score of 0.270071 is prioritized.|85. Despite a demographic score of 0.255025, the financial behavioral score of 0.271134 is more critical.|86. The demographic score is 0.363600, yet the financial behavioral score of 0.270862 takes precedence.|87. A demographic score of 0.366000 is overshadowed by the financial behavioral score of 0.271022.|88. The demographic score is 0.251495, but the financial behavioral score of 0.271147 is more significant.|89. With a demographic score of 0.363711, the financial behavioral score of 0.271023 is more important.|90. The demographic score is 0.362001, while the financial behavioral score of 0.271023 holds more weight.|91. A demographic score of 0.363746 is less important than the financial behavioral score of 0.270967.|92. The demographic score is 0.363697, but the financial behavioral score of 0.270380 is prioritized.|93. Despite a demographic score of 0.365417, the financial behavioral score of 0.270828 is more critical.|94. The demographic score is 0.363436, yet the financial behavioral score of 0.270832 takes precedence.|95. A demographic score of 0.364217 is overshadowed by the financial behavioral score of 0.270875.|96. The demographic score is 0.364001, but the financial behavioral score of 0.270141 is more significant.|97. With a demographic score of 0.363537, the financial behavioral score of 0.270118 is more important.|98. The demographic score is 0.359559, while the financial behavioral score of 0.270332 holds more weight.|99. A demographic score of 0.253454 is less important than the financial behavioral score of 0.270135.|100. The demographic score is 0.363798, but the financial behavioral score of 0.270024 is prioritized.|101. Despite a demographic score of 0.366435, the financial behavioral score of 0.270123 is more critical.|102. The demographic score is 0.363287, yet the financial behavioral score of 0.270145 takes precedence.|103. A demographic score of 0.363697 is overshadowed by the financial behavioral score of 0.270093.|104. The demographic score is 0.363788, but the financial behavioral score of 0.270127 is more significant.|105. With a demographic score of 0.363644, the financial behavioral score of 0.270127 is more important.|106. The demographic score is 0.363287, while the financial behavioral score of 0.267390 holds more weight.|107. A demographic score of 0.363287 is less important than the financial behavioral score of 0.267425.|108. The demographic score is 0.363391, but the financial behavioral score of 0.267460 is prioritized.|109. Despite a demographic score of 0.363391, the financial behavioral score of 0.267432 is more critical.|110. The demographic score is 0.363711, yet the financial behavioral score of 0.267432 takes precedence.|111. A demographic score of 0.366373 is overshadowed by the financial behavioral score of 0.267402.|112. The demographic score is 0.366373, but the financial behavioral score of 0.267425 is more significant.|113. With a demographic score of 0.365565, the financial behavioral score of 0.267460 is more important.|114. The demographic score is 0.366146, while the financial behavioral score of 0.267432 holds more weight.|115. A demographic score of 0.259022 is less important than the financial behavioral score of 0.267432.|116. The demographic score is 0.363350, but the financial behavioral score of 0.268752 is prioritized.|117. Despite a demographic score of 0.363159, the financial behavioral score of 0.269282 is more critical.|118. The demographic score is 0.365001, yet the financial behavioral score of 0.268884 takes precedence.|119. A demographic score of 0.363674 is overshadowed by the financial behavioral score of 0.268938.|120. The demographic score is 0.365006, but the financial behavioral score of 0.269041 is more significant.|121. With a demographic score of 0.363436, the financial behavioral score of 0.268798 is more important.|122. The demographic score is 0.361667, while the financial behavioral score of 0.269245 holds more weight.|123. A demographic score of 0.363632 is less important than the financial behavioral score of 0.268830.|124. The demographic score is 0.363697, but the financial behavioral score of 0.268846 is prioritized.|125. Despite a demographic score of 0.363711, the financial behavioral score of 0.269082 is more critical.|"}],
  temperature=0,
  top_p=1,
  max_tokens=8192,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

