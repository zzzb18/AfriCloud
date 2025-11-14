import requests
import json
import re


def main(spotNames, text):
    """
    查询POI详细信息并将text中的POI名称替换为模板内容

    Args:
        spotNames: POI名称列表，格式: ["地点1", "地点2", ...]
        text: 原始文本，包含POI名称

    Returns:
        包含渲染后HTML和POI详细信息的字典
    """
    poiResults = []
    allValid = 1  # 初始假设所有景点都能找到

    print(f"开始查询 {len(spotNames)} 个POI信息...")

    # 遍历每个地点进行查询
    for keyword in spotNames:
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
                resultData = response.json()
                items = resultData.get('body', [])

                if items:
                    bestPoi = None
                    # 优先寻找完全匹配的景点
                    for item in items:
                        if item.get("name") == keyword:
                            bestPoi = item
                            break
                    # 找不到则取第一个结果
                    if not bestPoi and items:
                        bestPoi = items[0]

                    if bestPoi:
                        # 清理None值
                        for attribute in bestPoi:
                            if bestPoi[attribute] is None:
                                bestPoi[attribute] = ""

                        poiResults.append({
                            "keyword": keyword,
                            "totalCount": len(items),
                            "poi": bestPoi
                        })
                        print(f"找到POI: {keyword} -> {bestPoi.get('name')}")
                    else:
                        print(f"关键词 '{keyword}' 未找到合适的POI信息")
                        allValid = 0
                else:
                    print(f"关键词 '{keyword}' 未找到POI信息")
                    allValid = 0
            else:
                errorMsg = f"请求失败，状态码: {response.status_code}"
                print(f"关键词 '{keyword}' {errorMsg}")
                allValid = 0

        except requests.exceptions.RequestException as e:
            errorMsg = f"请求异常: {str(e)}"
            print(f"关键词 '{keyword}' {errorMsg}")
            allValid = 0

    # 渲染模板并替换text中的POI名称
    modifiedText = replacePoiInText(text, poiResults)

    return {
        "result": modifiedText
    }


def replacePoiInText(text, poiResults):
    """
    将text中的POI名称替换为对应的HTML模板

    Args:
        text: 原始文本
        poiResults: POI查询结果列表

    Returns:
        替换后的文本
    """
    # 创建POI名称到模板的映射
    poiTemplates = {}

    for poiData in poiResults:
        poi = poiData.get("poi", {})
        keyword = poiData.get("keyword", "")

        # 提取需要的数据（根据实际API响应结构调整）
        name = poi.get("name", "未知名称")
        rating = poi.get("rating", "")  # 评分
        commentNumber = poi.get("commentNumber", 0)  # 评论数量
        image = poi.get("image", "")  # 图片URL
        businessHours = poi.get("businessHours", "")  # 营业时间
        playTime = poi.get("playTime", "")  # 建议游玩时间

        # 处理评论数量显示
        commentDisplay = f"{commentNumber}条评论" if commentNumber else "暂无评论"

        # 处理评分显示
        ratingDisplay = f"{rating}/5" if rating else "暂无评分"

        # POIid
        poiId = poi.get("poiId", 0)
        # 生成模板（根据实际数据结构调整）
        template = f'''
        <!-- POI卡片 -->
        <div style="display: flex; background-color: #ffffff; border-radius: 12px; padding: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border: 1px solid #f0f0f0; ">
            <div style="width: 100px; height: 80px; background-color: #f5f5f5; border-radius: 8px; margin-right: 12px; overflow: hidden; display: flex; align-items: center; justify-content: center;">
            <a href="https://cp.jegotrip.com.cn/partners/social/produce/socialh5/index.html#/poiDetails?poiId={poiId}&poiUid=&source=&wyx=1fb65a1e" target="_blank">
                <img src="{image}" alt="{name}" style="width: 100%; height: 100%; object-fit: cover;" onerror="this.style.display='none';this.parentNode.innerHTML='🏨';this.parentNode.style.display='flex;align-items:center;justify-content:center;color:#999;font-size:12px;'">
            </a>
            </div>
            <div style="flex: 1; margin-top:10px; ">
                <div style="font-size: 16px; font-weight: 600; color: #333; margin-top: -60px;">
                    {name}
                </div>
                <div style="display: flex; align-items: center; margin-top: -30px;">
                    <div style="color: #FF6B35; font-size: 14px; font-weight: 600; ">{ratingDisplay}</div>
                    <div style="font-size: 12px; color: #666; margin-left: 8px;">{commentDisplay}</div>
                </div>
                <div style="display: flex; font-size: 12px; color: #666; margin-top: -30px; display: flex">
                    <div style="font-weight: 500;">营业时间:</div> {businessHours or '暂无信息'}
                </div>
                <div style="display: flex; font-size: 12px; color: #666; margin-top: -30px;">
                    <div style="font-weight: 500;">建议游玩:</div> {playTime or '暂无信息'}
                </div>
                </div>
            </div>
        </div>
        '''

        poiTemplates[keyword] = template

    # 替换text中的POI名称
    modifiedText = text
    for keyword, template in poiTemplates.items():
        # 使用正则表达式进行精确匹配替换
        modifiedText = re.sub(
            r'(?<![a-zA-Z0-9])' + re.escape(keyword) + r'(?![a-zA-Z0-9])',
            template,
            modifiedText
        )

    return modifiedText
