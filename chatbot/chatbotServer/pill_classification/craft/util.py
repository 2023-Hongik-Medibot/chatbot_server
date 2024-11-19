import math
from copy import deepcopy
import copy
import numpy as np

def parse_region_map(reg_map, min_heat):
    # 큰 값을 가지는, 중심점이 될 후보들을 찾는다
    sum = np.sum(reg_map>min_heat)
    min_char_size = np.clip(int(sum/500),1,200) * 10
    center_poses = []
    for x_idx, y_axis in enumerate(reg_map):
        for y_idx, x_value in enumerate(y_axis):
            if reg_map[x_idx][y_idx] > min_heat:

                # 중심점 후보들의 거리가 가까우면, 둘중 큰 값만 남긴다
                if len(center_poses) == 0:
                    center_poses.append((x_idx, y_idx))
                else:
                    checked = False
                    for pre_saved in center_poses:
                        if math.sqrt((pre_saved[0] - x_idx) ** 2 + (pre_saved[1] - y_idx) ** 2) < min_char_size:
                            checked = True
                            if reg_map[x_idx][y_idx] > reg_map[pre_saved[0]][pre_saved[1]]:
                                center_poses.remove(pre_saved)
                                center_poses.append((x_idx, y_idx))

                            else:
                                break
                    if checked is False:
                        center_poses.append((x_idx, y_idx))
    # Box 그리기
    boxes = []
    for center in center_poses:
        cur_x = center[0]
        cur_y = center[1]

        width_left, width_right, height_up, height_down = 1, 1, 1,1 
        d = 1
        while True:
            if reg_map[np.clip(cur_x - width_left,0,223)][cur_y] < reg_map[np.clip(cur_x - width_left + d,0,cur_x)][cur_y]:
                width_left += 1
            else:
                break

        while True:
            if reg_map[np.clip(cur_x + width_right,0,223)][cur_y] < reg_map[np.clip(cur_x + width_right - d,cur_x,223)][cur_y]:
                width_right += 1
            else:
                break

        while True:
            if reg_map[cur_x][np.clip(cur_y + height_up,0,223)] < reg_map[cur_x][np.clip(cur_y + height_up - d,cur_y,223)]:
                height_up += 1
            else:
                break

        while True:
            if reg_map[cur_x][np.clip(cur_y - height_down,0,223)] < reg_map[cur_x][np.clip(cur_y - height_down + d,0,cur_y)]:
                height_down += 1
            else:
                break
          #########    위                 왼                   아래                  오
        boxes.append([cur_x - width_left, cur_y - height_down, cur_x + width_right, cur_y + height_up])
    # Box 들의 겹치는 영역 많으면 하나 제거
    unique_boxes = deepcopy(boxes)
    overlapped = []
    removed_list=[]
    for i, unique_box in enumerate(boxes):
        for j, box in enumerate(boxes):
            if j in removed_list:
              continue
            if i in removed_list:
              break
            if i>=j:
                pass
            else:
                x_right = min(unique_box[2], box[2]) ## 아래
                x_left = max(unique_box[0], box[0]) ### 위
                y_top = min(unique_box[3], box[3]) ####오
                y_bot = max(unique_box[1], box[1])###왼

                if x_right <= x_left:
                    pass
                elif y_top <= y_bot:
                    pass
                else:
                    i_area = (boxes[i][3] - boxes[i][1])*(boxes[i][2] - boxes[i][0])
                    j_area = (boxes[j][3] - boxes[j][1])*(boxes[j][2] - boxes[j][0])
                    if i_area >= j_area:
                      unique_area = j_area
                      remove_box = boxes[j]
                      num = j
                    else :
                      unique_area = i_area
                      remove_box = boxes[i]
                      num = i
                    overlap_area = (x_right - x_left) * (y_top - y_bot)
                    ratio = float(overlap_area / unique_area)
                    if ratio > 0.7:
                        try:
                            unique_boxes.remove(remove_box)
                            overlapped.append(remove_box)
                            removed_list.append(num)
                            if num == i :
                              break
                        except:
                            pass

    ##### 경계 잇기
    boxes = copy.deepcopy(unique_boxes)
    connected_boxes = []
    for num1,box in enumerate(boxes):
      for num2 in range(num1+1,len(boxes)):
        u1,d1,l1,r1 = boxes[num1][0],boxes[num1][2],boxes[num1][1],boxes[num1][3]
        u2,d2,l2,r2 = boxes[num2][0],boxes[num2][2],boxes[num2][1],boxes[num2][3]
        if abs(l1-l2)<1 and abs(r1-r2) <1:
          ####서로 아예 붙지 않음
          if d1 != u2 or d2!= u1:
            pass
          #### 겹침
          if (u1 < u2 and d1<d2) or(u2 < u1 and d2<d1):
            connected_boxes.append((num1,num2))
          ###서로 붙음
          if d2==u1 or d1 == u2:
            connected_boxes.append((num1,num2))
          ###완전 같거나 포함됨
          if (u1 <= u2 and d1 == d2) or (u2 <= u1 and d1 == d2) or (u1 ==u2 and d1 <= d2) or (u1 ==u2 and d2 <= d1):
            connected_boxes.append((num1,num2))
    while True:
      union_box = []
      for i in range(len(boxes)):
        uni = []
        for set_box in (connected_boxes):
          if i in set_box:
            uni.append(set_box[0])
            uni.append(set_box[1])
        uni = set(uni)
        if len(uni)!=0:
          union_box.append(tuple(uni))
      if len(connected_boxes) == len(set(union_box)):
        connected_boxes = list(set(union_box))
        break
      connected_boxes = set(union_box)
    new_boxes = copy.deepcopy(boxes)
    for box in list(connected_boxes):
      up,down = 1000, 0
      for num in box:
        ##위가
        if boxes[num][0] <up:
          up = boxes[num][0]
        if boxes[num][2] > down:
          down = boxes[num][2]
        new_boxes.remove(boxes[num])
      new_boxes.append([up,boxes[num][1],down,boxes[num][3]])
    return new_boxes
