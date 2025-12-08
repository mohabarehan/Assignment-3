# Проект: Классификация самолетов (Planes Dataset Project)
## Описание проекта
#Этот проект предназначен для классификации изображений самолетов на **гражданские** и **военные**.  
#Проект использует модель **CLIP** для "Zero-Shot" классификации, что позволяет определять тип самолета без дополнительного обучения.  
#Изображения размещены в главной папке `planes_dataset`, с подкаталогами:  
#- `0` → изображения гражданских самолетов  
#- `1` → изображения военных самолетов  
#Скрипт проверяет каждое изображение, определяет его тип и сохраняет полный отчет.


##Результаты#
#Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.
#0/planes_1.jpg.jpg → civilian airplane (90.46%)
#0/planes_10.jpg.jpg → civilian airplane (93.16%)
#0/planes_11.jpg.jpg → civilian airplane (86.07%)
#0/planes_12.jpg.jpg → civilian airplane (92.49%)
#0/planes_13.jpg.jpg → civilian airplane (85.06%)
#0/planes_14.jpg.jpg → civilian airplane (93.30%)
#0/planes_15.jpg.jpg → civilian airplane (79.01%)
#0/planes_2.jpg.jpg → civilian airplane (89.74%)
#0/planes_3.jpg.jpg → civilian airplane (96.44%)
#0/planes_4.jpg.jpg → civilian airplane (71.88%)
#0/planes_5.jpg.jpg → civilian airplane (85.97%)
#0/planes_6.jpg.jpg → civilian airplane (61.89%)
#0/planes_7.jpg.jpg → civilian airplane (74.77%)
#0/planes_8.jpg.jpg → civilian airplane (83.05%)
#0/planes_9.jpg.jpg → civilian airplane (71.70%)
#1/planes_16.jpg.jpg → military airplane (80.14%)
#1/planes_17.jpg.jpg → military airplane (64.04%)
#1/planes_18.jpg.jpg → military airplane (73.11%)
#1/planes_19.jpg.jpg → military airplane (52.72%)
#1/planes_20.jpg.jpg → military airplane (64.19%)
#1/planes_21.jpg.jpg → military airplane (77.74%)
#1/planes_22.jpg.jpg → military airplane (67.05%)
#1/planes_23.jpg.jpg → military airplane (57.71%)
#1/planes_24.jpg.jpg → military airplane (58.55%)
#1/planes_25.jpg.jpg → military airplane (73.65%)
#1/planes_26.jpg.jpg → military airplane (79.50%)
#1/planes_27.jpg.jpg → military airplane (75.83%)
#1/planes_28.jpg.jpg → military airplane (85.85%)
#1/planes_29.jpg.jpg → military airplane (71.46%)
#1/planes_30.jpg.jpg → military airplane (72.89%)

#==========================
#Total images processed: 30
#Civilian airplanes: 15
#Military airplanes: 15
#==========================
#CLIP Zero-Shot Accuracy: 30/30 = 100.00%
#ViT ImageNet Accuracy:  30/30 = 100.00%
#==========================