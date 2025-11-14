import requests
import json


def main(dailyArrangement, conversationId, user):
    """
    从dailyArrangement中提取POI信息，调用地图接口生成H5图片标签

    Args:
        dailyArrangement: main函数返回的dailyArrangement字段

    Returns:
        包含result和logInfo字段的字典
    """

    logInfo = []  # 用于存储日志信息
    result = ""  # 用于存储最终结果

    if dailyArrangement.get("enableMap") != 1:
        logInfo.append("enableMap不为1，跳过地图生成")
        return {
            "result": result,
            "logInfo": logInfo
        }

    try:
        # 提取所有POI信息
        poiInfos = []
        logInfo.append("开始提取POI信息")

        # 遍历每天的安排
        for dayIndex, dayData in enumerate(dailyArrangement.get("arrangement", [])):
            dayPoiCount = 0
            for poiInfo in dayData.get("poiInfo", []):
                poi = poiInfo.get("poi", {})
                if poi and "lon" in poi and "lat" in poi:
                    # 确保经纬度转换为字符串
                    lonStr = str(poi["lon"]) if poi["lon"] is not None else ""
                    latStr = str(poi["lat"]) if poi["lat"] is not None else ""
                    poiName = str(poi.get("name", "")).strip()
                    poiInfos.append({
                        "poiName": poiName,
                        "lon": lonStr,
                        "lat": latStr
                    })
                    dayPoiCount += 1
            logInfo.append(f"第{dayIndex + 1}天提取到{dayPoiCount}个有效POI")

        if not poiInfos:
            logInfo.append("未找到有效的POI信息，无法生成地图")
            return {
                "result": result,
                "logInfo": logInfo
            }

        logInfo.append(f"总共提取到{len(poiInfos)}个POI信息")

        # 构建请求参数
        mapPayload = {
            "poiInfos": poiInfos,
            "conversationId": conversationId,
            "user": user
        }

        logInfo.append(f"发送给地图接口的数据: {json.dumps(mapPayload, ensure_ascii=False)}")

        # 设置请求头
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
        }

        # 调用地图接口
        logInfo.append("开始调用地图接口")
        response = requests.post(
            "https://topenapi.yidingbao.shop/api/assistant/message/map",
            headers=headers,
            data=json.dumps(mapPayload),
            timeout=15
        )

        # 处理响应
        if response.status_code == 200:
            resultData = response.json()
            logInfo.append(f"地图接口返回: {json.dumps(resultData, ensure_ascii=False)}")

            imageUrl = resultData.get('data', {}).get('image')
            if imageUrl:
                # 生成H5图片标签
                result = f'<img src="{imageUrl}" alt="行程地图" style="width: 100%; height: auto;">'
                logInfo.append("成功生成地图图片标签")
            else:
                logInfo.append("地图接口返回数据中未找到image字段")
        else:
            error_msg = f"地图接口请求失败，状态码: {response.status_code}"
            logInfo.append(error_msg)

    except requests.exceptions.Timeout:
        logInfo.append("地图接口请求超时")
    except requests.exceptions.ConnectionError:
        logInfo.append("地图接口连接错误")
    except requests.exceptions.RequestException as e:
        logInfo.append(f"地图接口请求异常: {str(e)}")
    except Exception as e:
        logInfo.append(f"程序执行异常: {str(e)}")

    return {
        "result": result,
        "logInfo": logInfo
    }