import requests
import json


def main(daily_arrangement):
    """
    从dailyArrangement中提取POI信息，调用地图接口生成H5图片标签

    Args:
        daily_arrangement: main函数返回的dailyArrangement字段

    Returns:
        H5图片标签字符串，如果失败返回错误信息
    """

    if daily_arrangement.get("enableMap") != 1:
        return {
            "result": ""
        }
    try:
        # 提取所有POI信息
        poi_infos = []

        # 遍历每天的安排
        for day_data in daily_arrangement.get("arrangement", []):
            for poi_info in day_data.get("poi_info", []):
                poi = poi_info.get("poi", {})
                if poi and "lon" in poi and "lat" in poi:
                    # 确保经纬度转换为字符串
                    lon_str = str(poi["lon"]) if poi["lon"] is not None else ""
                    lat_str = str(poi["lat"]) if poi["lat"] is not None else ""
                    poi_name = str(poi.get("name", "")).strip()
                    poi_infos.append({
                        "poiName": poi_name,
                        "lon": lon_str,
                        "lat": lat_str
                    })

        if not poi_infos:
            return {"result": "<p>未找到有效的POI位置信息</p>"}

        # 构建请求参数
        map_payload = {
            "poiInfos": poi_infos
        }

        print(f"发送给地图接口的数据: {json.dumps(map_payload, ensure_ascii=False)}")

        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        }

        # 调用地图接口
        response = requests.post(
            "https://topenapi.yidingbao.shop/api/assistant/message/map",
            headers=headers,
            data=json.dumps(map_payload),
            timeout=15
        )

        # 处理响应
        if response.status_code == 200:
            result_data = response.json()
            print(f"地图接口返回: {json.dumps(result_data, ensure_ascii=False)}")
            image_url = result_data.get('data', {}).get('image')
            if image_url:
                # 生成H5图片标签
                h5_tag = f'<img src="{image_url}" alt="行程地图" style="width: 100%; height: auto;">'
                return {"result": h5_tag}
            else:
                return {
                    "result": ""
                }
        else:
            return {"result": ""}

    except requests.exceptions.RequestException as e:
        return {"result": ""}
    except Exception as e:
        return {"result": ""}
