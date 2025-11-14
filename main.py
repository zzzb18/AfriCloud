import requests
import json


def f(daily_arr):
    """
    按日期查询POI详细信息，直接整合查询逻辑

    Args:
        daily_arr: 按日期组织的POI名称列表
                  格式: [{"locations": ["地点1", "地点2"]}, ...]

    Returns:
        按原日期结构组织的POI详细信息，包含enableMap字段
    """
    daily_results = []
    all_days_valid = True  # 初始假设所有天的景点都能找到

    # 遍历每一天的地点列表
    for day_index, day_data in enumerate(daily_arr):
        locations = day_data.get("locations", [])
        if not locations:
            print(f"第 {day_index + 1} 天没有地点信息，跳过查询")
            daily_results.append({"day": day_index + 1, "locations": []})
            all_days_valid = False  # 没有地点信息，设为无效
            continue

        print(f"开始查询第 {day_index + 1} 天的POI信息...")
        day_poi_results = []
        day_valid = True  # 假设当天所有景点都能找到

        # 遍历当天每个地点进行查询
        for keyword in locations:
            # 构建请求参数
            payload = {
                "poiNames": [keyword],
                "pageSize": 10,
                "pageNum": 1,
                "dataSource": "内容组",
                "showFields": ["poiExtends"],
                "queryBizFlag": 1
            }

            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
            }

            try:
                # 发送POST请求
                response = requests.post(
                    "https://topenapi.yidingbao.shop/api/poi/list",
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=10
                )

                # 处理响应
                if response.status_code == 200:
                    result_data = response.json()
                    items = result_data.get('body', [])

                    if items:
                        best_poi = None
                        # 优先寻找完全匹配的景点
                        for item in items:
                            if item.get("name") == keyword and item.get("tagMain") == "景点":
                                best_poi = item
                                break
                        # 找不到则寻找第一个景点
                        if not best_poi:
                            for item in items:
                                if item.get("tagMain") == "景点":
                                    best_poi = item
                                    break
                        # 再找不到则取第一个结果
                        if not best_poi and items:
                            best_poi = items[0]

                        if best_poi:
                            for attribute in best_poi:
                                if best_poi[attribute] is None:
                                    best_poi[attribute] = ""
                            day_poi_results.append({
                                "keyword": keyword,
                                "totalCount": len(items),
                                "poi": best_poi
                            })
                        else:
                            print(f"关键词 '{keyword}' 未找到合适的POI信息")
                            day_valid = False  # 当天有一个景点没找到
                    else:
                        print(f"关键词 '{keyword}' 未找到POI信息")
                        day_valid = False  # 当天有一个景点没找到
                else:
                    error_msg = f"请求失败，状态码: {response.status_code}"
                    print(f"关键词 '{keyword}' {error_msg}")
                    day_valid = False  # 当天有一个景点请求失败

            except requests.exceptions.RequestException as e:
                error_msg = f"请求异常: {str(e)}"
                print(f"关键词 '{keyword}' {error_msg}")
                day_valid = False  # 当天有一个景点请求异常

        # 如果当天有任何景点没找到，设置all_days_valid为False
        if not day_valid:
            all_days_valid = False

        # 添加当天结果到总结果
        daily_results.append({
            "day": day_index + 1,
            "original_locations": locations,
            "poi_info": day_poi_results
        })

        result = {
            "dailyArrangement": {
                "arrangement": daily_results,
                "enableMap": all_days_valid
            }
        }
        print(result)
    return json.loads(json.dumps(result, ensure_ascii=False))
    # 使用json.dumps确保标准JSON格式
if __name__ == '__main__':
    f([
    {
      "locations": [
        "世界之窗",
        "华侨城创意文化园"
      ]
    },
    {
      "locations": [
        "大梅沙海滨公园",
        "中英街"
      ]
    }
  ])