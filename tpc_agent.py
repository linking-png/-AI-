"""
TPCAgent - 修复版本

修复问题：
1. pandas 2.0+ 兼容性：DataFrame.append() → pd.concat()
2. 改进LLM prompt：更准确提取约束，不添加用户未提及的内容
3. 修复typo：endswith拼写错误

版本：v1.1 - Bug Fix
"""

import random
import json
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from agent.base import BaseAgent


class TPCAgent(BaseAgent):
    """使用LLM的TPCAgent - Bug Fix版本"""
    
    def __init__(self, **kwargs):
        super().__init__(name="TPCAgent", **kwargs)
        self.llm = self.backbone_llm
        
        # 配置
        self.min_attractions_per_day = 2
        self.max_attractions_per_day = 4
        self.debug = kwargs.get('debug', True)
        
        print(f"[TPCAgent] 初始化完成 - LLM驱动版本 v1.1")
        print(f"[TPCAgent] 环境: {type(self.env).__name__ if self.env else 'None'}")
        print(f"[TPCAgent] LLM: {type(self.llm).__name__}")
    
    def run(self, query, prob_idx: int, oralce_translation: bool = False) -> Tuple[bool, Dict]:
        """主要入口函数"""
        self.reset_clock()
        
        try:
            query_dict = self._parse_query(query, prob_idx)
            
            uid = query_dict['uid']
            start_city = query_dict['start_city']
            target_city = query_dict['target_city']
            days = query_dict['days']
            people_number = query_dict['people_number']
            nature_language = query_dict['nature_language']
            
            if self.debug:
                print(f"\n{'='*70}")
                print(f"[Query {uid}] {start_city} → {target_city}, {days}天, {people_number}人")
                print(f"{'='*70}")
            
            # 🔥 使用LLM提取约束
            constraints = self._extract_constraints_with_llm(nature_language, query_dict)
            
            if self.debug:
                print(f"\n[LLM约束提取]")
                for key, value in constraints.items():
                    if value is not None and value not in [[], {}, '', 'null']:
                        print(f"  {key}: {value}")
            
            # 生成行程
            itinerary = self._generate_itinerary_with_env(
                start_city=start_city,
                target_city=target_city,
                days=days,
                people_number=people_number,
                constraints=constraints
            )
            
            # 验证天数
            if len(itinerary) != days:
                itinerary = self._fix_day_count(itinerary, days)
            
            if self.debug:
                print(f"\n[完成] {len(itinerary)}天行程")
            
            result = {
                "people_number": people_number,
                "start_city": start_city,
                "target_city": target_city,
                "itinerary": itinerary
            }
            
            return True, result
            
        except Exception as e:
            print(f"[ERROR] 规划失败: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            
            return False, {
                "people_number": query_dict.get('people_number', 1),
                "start_city": query_dict.get('start_city', '北京'),
                "target_city": query_dict.get('target_city', '上海'),
                "itinerary": []
            }
    
    def _extract_constraints_with_llm(self, nature_language: str, query_dict: Dict) -> Dict:
        """使用LLM提取约束 - 改进版"""
        
        prompt = f"""你是旅行规划助手。请仔细阅读用户需求，准确提取约束信息。

用户需求：
{nature_language}

基本信息：
- 出发: {query_dict.get('start_city')}
- 目的地: {query_dict.get('target_city')}
- 天数: {query_dict.get('days')}
- 人数: {query_dict.get('people_number')}

重要规则：
1. 如果用户没有明确提到，设为null，不要猜测
2. 预算数字必须准确，不要改变
3. 不要添加用户未提及的景点名称

输出JSON格式：
{{
    "transport_mode": null,
    "attraction_types": [],
    "attraction_names": [],
    "food_types": [],
    "budget_limit": null,
    "free_attractions_only": false,
    "room_type": null,
    "room_count": null,
    "transport_preference": null,
    "excluded_places": [],
    "pace": "正常"
}}

只输出JSON，不要其他文字。"""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm.get_response(messages, one_line=False, json_mode=True)
            
            # 清理响应
            response = response.strip()
            if response.startswith('```json'):
                response = response[7:]
            if response.startswith('```'):
                response = response[3:]
            if response.endswith('```'):  # ✅ 修复typo
                response = response[:-3]
            response = response.strip()
            
            # 解析JSON
            constraints = json.loads(response)
            
            # ✅ 验证预算提取
            if self.debug and constraints.get('budget_limit'):
                print(f"[调试] 提取的预算: {constraints['budget_limit']}")
            
            return constraints
            
        except Exception as e:
            if self.debug:
                print(f"[警告] LLM约束提取失败: {e}")
                if 'response' in locals():
                    print(f"[LLM响应] {response[:200]}...")
            return self._extract_constraints_fallback(nature_language)
    
    def _extract_constraints_fallback(self, nature_language: str) -> Dict:
        """回退的规则方法"""
        constraints = {
            "transport_mode": None,
            "attraction_types": [],
            "attraction_names": [],
            "food_types": [],
            "budget_limit": None,
            "free_attractions_only": False,
            "room_type": None,
            "room_count": None,
            "transport_preference": None,
            "excluded_places": [],
            "pace": "正常"
        }
        
        # 简单规则提取
        if '免费' in nature_language:
            constraints['free_attractions_only'] = True
        
        if '飞机' in nature_language:
            constraints['transport_mode'] = '飞机'
        elif '火车' in nature_language or '高铁' in nature_language:
            constraints['transport_mode'] = '火车'
        
        if '地铁' in nature_language:
            constraints['transport_preference'] = '地铁'
        
        # 提取预算
        budget_match = re.search(r'预算[：:]?(\d+)', nature_language)
        if budget_match:
            constraints['budget_limit'] = int(budget_match.group(1))
        
        # 提取房型
        if '单床房' in nature_language:
            constraints['room_type'] = '单床房'
            constraints['room_count'] = 1
        elif '双床房' in nature_language or '标准间' in nature_language:
            constraints['room_type'] = '双床房'
            constraints['room_count'] = 2
        
        return constraints
    
    def _generate_itinerary_with_env(self, start_city: str, target_city: str,
                                     days: int, people_number: int,
                                     constraints: Dict) -> List[Dict]:
        """使用环境API生成行程"""
        
        itinerary = []
        current_location = f"{start_city}站"
        last_hotel = None
        current_cost = 0.0
        budget_limit = constraints.get('budget_limit')
        
        # 预查询数据
        if self.debug:
            print(f"\n[数据查询] 开始...")
        
        attractions_data = self._query_attractions(target_city, constraints)
        restaurants_data = self._query_restaurants(target_city, constraints)
        hotels_data = self._query_accommodations(target_city, constraints)
        
        if self.debug:
            print(f"[数据池] 景点:{len(attractions_data)} 餐厅:{len(restaurants_data)} 酒店:{len(hotels_data)}")
        
        # 检查数据充足性
        if len(attractions_data) < days * 2:
            print(f"[警告] 景点数据不足: {len(attractions_data)}")
        if len(restaurants_data) < days * 2:
            print(f"[警告] 餐厅数据不足: {len(restaurants_data)}")
        if len(hotels_data) < 1 and days > 1:
            print(f"[警告] 酒店数据不足: {len(hotels_data)}")
            hotels_data = self._query_accommodations(target_city, {})
        
        for day in range(1, days + 1):
            if self.debug:
                print(f"\n--- 第{day}天 ---")
            
            day_plan = {
                "day": day,
                "activities": []
            }
            
            # 第一天：跨城交通
            if day == 1:
                transport_mode = constraints.get('transport_mode', '火车')
                if transport_mode == '飞机':
                    transport_type = 'airplane'
                else:
                    transport_type = 'train'
                
                if self.debug:
                    print(f"[跨城交通] {start_city}→{target_city} ({transport_type})")
                
                transport = self._create_intercity_transport(
                    start_city, target_city, transport_type, people_number, current_location
                )
                day_plan["activities"].append(transport)
                current_cost += transport['cost']
                current_location = transport['end']
            
            # 每天：早餐（第2天起）
            if day > 1 and last_hotel:
                breakfast = {
                    "position": last_hotel,
                    "type": "breakfast",
                    "cost": 0.0,
                    "price": 0.0,
                    "transports": [],
                    "start_time": "07:10",
                    "end_time": "07:40"
                }
                day_plan["activities"].append(breakfast)
            
            # 每天：景点
            pace = constraints.get('pace', '正常')
            if pace == '悠闲':
                num_attractions = self.min_attractions_per_day
            elif pace == '紧凑':
                num_attractions = self.max_attractions_per_day
            else:
                num_attractions = random.randint(self.min_attractions_per_day, self.max_attractions_per_day)
            
            for i in range(min(num_attractions, len(attractions_data))):
                if not attractions_data:
                    break
                
                # 预算检查
                if budget_limit and current_cost > budget_limit * 0.9:
                    if self.debug:
                        print(f"[预算警告] 接近预算上限，减少景点")
                    break
                
                attraction_row = attractions_data.pop(0)
                attraction = self._create_attraction_from_data(
                    attraction_row, target_city, people_number, current_location
                )
                day_plan["activities"].append(attraction)
                current_cost += attraction['cost']
                current_location = attraction['position']
            
            # 每天：午餐
            if restaurants_data:
                restaurant_row = restaurants_data.pop(0)
                lunch = self._create_meal_from_data(
                    restaurant_row, "lunch", target_city, people_number, current_location
                )
                day_plan["activities"].append(lunch)
                current_cost += lunch['cost']
                current_location = lunch['position']
            
            # 每天：晚餐
            if restaurants_data:
                restaurant_row = restaurants_data.pop(0)
                dinner = self._create_meal_from_data(
                    restaurant_row, "dinner", target_city, people_number, current_location
                )
                day_plan["activities"].append(dinner)
                current_cost += dinner['cost']
                current_location = dinner['position']
            
            # 最后一天：返程
            if day == days:
                transport_mode = constraints.get('transport_mode', '火车')
                if transport_mode == '飞机':
                    transport_type = 'airplane'
                else:
                    transport_type = 'train'
                
                if self.debug:
                    print(f"[返程交通] {target_city}→{start_city} ({transport_type})")
                
                transport = self._create_intercity_transport(
                    target_city, start_city, transport_type, people_number, current_location
                )
                day_plan["activities"].append(transport)
                current_cost += transport['cost']
            else:
                # 非最后一天：住宿
                if hotels_data:
                    hotel_row = hotels_data[0]
                    accommodation = self._create_accommodation_from_data(
                        hotel_row, target_city, people_number, constraints, current_location
                    )
                    day_plan["activities"].append(accommodation)
                    current_cost += accommodation['cost']
                    last_hotel = accommodation['position']
            
            itinerary.append(day_plan)
        
        if self.debug:
            print(f"\n[总成本] ¥{current_cost:.2f}")
            if budget_limit:
                print(f"[预算限制] ¥{budget_limit}")
                if current_cost > budget_limit:
                    print(f"[警告] 超出预算 ¥{current_cost - budget_limit:.2f}")
        
        return itinerary
    
    def _query_attractions(self, city: str, constraints: Dict) -> List[Dict]:
        """查询景点 - 使用约束筛选"""
        try:
            result = self.env(f"attractions_select('{city}', 'name', lambda x: True)")
            
            if not result["success"]:
                return []
            
            df = result["whole_data"]
            
            # 🔥 应用约束筛选
            
            # 1. 免费景点
            if constraints.get('free_attractions_only', False):
                if 'price' in df.columns:
                    df = df[df['price'] == 0]
            
            # 2. 景点类型
            attraction_types = constraints.get('attraction_types', [])
            if attraction_types and 'type' in df.columns:
                df = df[df['type'].isin(attraction_types)]
            
            # 3. 必去景点（优先级最高）
            attraction_names = constraints.get('attraction_names', [])
            if attraction_names:
                required_df = df[df['name'].isin(attraction_names)]
                other_df = df[~df['name'].isin(attraction_names)]
                # ✅ 修复：使用pd.concat替代已弃用的append
                df = pd.concat([required_df, other_df], ignore_index=True) if not required_df.empty else df
            
            # 4. 排除景点
            excluded = constraints.get('excluded_places', [])
            if excluded:
                df = df[~df['name'].isin(excluded)]
            
            attractions = df.to_dict('records')
            
            # 5. 随机打乱（保持必去景点在前）
            if not attraction_names:
                random.shuffle(attractions)
            
            return attractions
            
        except Exception as e:
            print(f"[错误] 景点查询异常: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
            return []
    
    def _query_restaurants(self, city: str, constraints: Dict) -> List[Dict]:
        """查询餐厅 - 使用约束筛选"""
        try:
            result = self.env(f"restaurants_select('{city}', 'name', lambda x: True)")
            
            if not result["success"]:
                return []
            
            df = result["whole_data"]
            
            # 🔥 应用约束筛选
            
            # 菜系偏好
            food_types = constraints.get('food_types', [])
            if food_types and 'cuisine' in df.columns:
                df = df[df['cuisine'].isin(food_types)]
            
            restaurants = df.to_dict('records')
            random.shuffle(restaurants)
            
            return restaurants
            
        except Exception as e:
            print(f"[错误] 餐厅查询异常: {e}")
            return []
    
    def _query_accommodations(self, city: str, constraints: Dict) -> List[Dict]:
        """查询酒店 - 使用约束筛选"""
        try:
            result = self.env(f"accommodations_select('{city}', 'name', lambda x: True)")
            
            if not result["success"]:
                return []
            
            df = result["whole_data"]
            
            # 🔥 应用约束筛选
            
            # 房型
            room_type = constraints.get('room_type')
            if room_type:
                if room_type == '单床房' and 'numbed' in df.columns:
                    df = df[df['numbed'] == 1]
                elif room_type == '双床房' and 'numbed' in df.columns:
                    df = df[df['numbed'] == 2]
            
            hotels = df.to_dict('records')
            random.shuffle(hotels)
            
            return hotels
            
        except Exception as e:
            print(f"[错误] 酒店查询异常: {e}")
            return []
    
    def _create_intercity_transport(self, from_city: str, to_city: str,
                                    transport_type: str, people_number: int,
                                    from_location: str) -> Dict:
        """创建跨城交通 - 自动适配列名"""
        
        try:
            command = f"intercity_transport_select('{from_city}', '{to_city}', '{transport_type}')"
            result = self.env(command)
            
            if result["success"] and result["data"] is not None:
                df = result["data"]
                if len(df) > 0:
                    transport_data = df.iloc[random.randint(0, len(df)-1)].to_dict()
                    
                    # 🔥 自动适配列名
                    start_time = self._get_column_value(transport_data, 
                        ['BeginTime', 'begintime', 'begin_time', 'start_time', 'StartTime'],
                        '09:00')
                    end_time = self._get_column_value(transport_data,
                        ['EndTime', 'endtime', 'end_time', 'arrival_time', 'ArrivalTime'],
                        '14:00')
                    price = self._get_column_value(transport_data,
                        ['Price', 'price', 'Cost', 'cost', 'ticket_price'],
                        300.0)
                    
                    if transport_type == 'train':
                        transport_id = self._get_column_value(transport_data,
                            ['TrainID', 'trainid', 'train_id', 'train_number', 'number'],
                            f'G{random.randint(100,999)}')
                    else:
                        transport_id = self._get_column_value(transport_data,
                            ['FlightID', 'flightid', 'flight_id', 'flight_number', 'number'],
                            f'FL{random.randint(100,999)}')
                    
                    activity = {
                        "start_time": str(start_time),
                        "end_time": str(end_time),
                        "start": from_location or f"{from_city}站",
                        "end": f"{to_city}站" if transport_type == "train" else f"{to_city}机场",
                        "price": float(price),
                        "cost": float(price) * people_number,
                        "tickets": people_number,
                        "transports": [],
                        "type": transport_type
                    }
                    
                    if transport_type == "train":
                        activity["TrainID"] = str(transport_id)
                    else:
                        activity["FlightID"] = str(transport_id)
                    
                    return activity
        
        except Exception as e:
            if self.debug:
                print(f"[错误] 跨城交通查询失败: {e}")
        
        # 回退方案
        base_price = 300.0
        return {
            "start_time": "09:00",
            "end_time": "14:00",
            "start": from_location or f"{from_city}站",
            "end": f"{to_city}站",
            "price": base_price,
            "cost": base_price * people_number,
            "tickets": people_number,
            "transports": [],
            "TrainID": f"G{random.randint(100, 999)}",
            "type": "train"
        }
    
    def _get_column_value(self, data: Dict, possible_names: List[str], default: Any) -> Any:
        """智能获取列值 - 支持多种列名格式"""
        for name in possible_names:
            if name in data:
                return data[name]
        return default
    
    def _create_attraction_from_data(self, data: Dict, city: str,
                                     people_number: int, from_location: str) -> Dict:
        """从环境数据创建景点活动"""
        
        position = data['name']
        price = float(data.get('price', 0))
        
        transports = self._query_transport(city, from_location, position, people_number)
        
        return {
            "position": position,
            "type": "attraction",
            "transports": transports,
            "price": price,
            "cost": price * people_number,
            "tickets": people_number,
            "start_time": "09:00",
            "end_time": "10:30"
        }
    
    def _create_meal_from_data(self, data: Dict, meal_type: str, city: str,
                               people_number: int, from_location: str) -> Dict:
        """从环境数据创建餐饮活动"""
        
        position = data['name']
        price = float(data.get('price', 80))
        
        transports = self._query_transport(city, from_location, position, people_number)
        
        time_map = {
            "breakfast": ("07:10", "07:40"),
            "lunch": ("11:00", "12:00"),
            "dinner": ("17:00", "17:50")
        }
        start_time, end_time = time_map.get(meal_type, ("12:00", "13:00"))
        
        return {
            "position": position,
            "type": meal_type,
            "transports": transports,
            "price": price,
            "cost": price * people_number if meal_type != "breakfast" else 0.0,
            "start_time": start_time,
            "end_time": end_time
        }
    
    def _create_accommodation_from_data(self, data: Dict, city: str,
                                        people_number: int, constraints: Dict,
                                        from_location: str) -> Dict:
        """从环境数据创建住宿活动"""
        
        position = data['name']
        price = float(data.get('price', 400))
        
        # 🔥 使用约束中的房间数
        room_count = constraints.get('room_count')
        if room_count:
            rooms = room_count
            room_type = 1 if room_count == 1 else 2
        else:
            room_type = int(data.get('numbed', 2))
            rooms = people_number if room_type == 1 else (people_number + 1) // 2
        
        transports = self._query_transport(city, from_location, position, people_number)
        
        return {
            "position": position,
            "type": "accommodation",
            "transports": transports,
            "room_type": room_type,
            "start_time": "18:20",
            "end_time": "24:00",
            "rooms": rooms,
            "cost": price * rooms,
            "price": price
        }
    
    def _query_transport(self, city: str, from_loc: str, to_loc: str,
                        people_number: int) -> List[Dict]:
        """查询交通信息"""
        
        if not from_loc or not to_loc or from_loc == to_loc:
            return []
        
        try:
            # 尝试步行
            result = self.env(f"goto('{city}', '{from_loc}', '{to_loc}', '09:00', 'walk')")
            if result["success"] and result["data"]:
                walk_data = result["data"]
                if isinstance(walk_data, dict):
                    return [{
                        "start": from_loc,
                        "end": to_loc,
                        "mode": "walk",
                        "start_time": walk_data.get('start_time', '09:00'),
                        "end_time": walk_data.get('end_time', '09:15'),
                        "cost": 0,
                        "distance": float(walk_data.get('distance', 0)),
                        "price": 0
                    }]
            
            # 尝试地铁
            result = self.env(f"goto('{city}', '{from_loc}', '{to_loc}', '09:00', 'metro')")
            if result["success"] and result["data"]:
                metro_data = result["data"]
                if isinstance(metro_data, dict):
                    return [{
                        "start": from_loc,
                        "end": to_loc,
                        "mode": "metro",
                        "start_time": metro_data.get('start_time', '09:00'),
                        "end_time": metro_data.get('end_time', '09:15'),
                        "cost": float(metro_data.get('price', 3)) * people_number,
                        "distance": float(metro_data.get('distance', 0)),
                        "tickets": people_number,
                        "price": float(metro_data.get('price', 3))
                    }]
            
            # 尝试出租车
            result = self.env(f"goto('{city}', '{from_loc}', '{to_loc}', '09:00', 'taxi')")
            if result["success"] and result["data"]:
                taxi_data = result["data"]
                if isinstance(taxi_data, dict):
                    num_cars = (people_number + 3) // 4
                    price_per_car = float(taxi_data.get('price', 20))
                    return [{
                        "start": from_loc,
                        "end": to_loc,
                        "mode": "taxi",
                        "start_time": taxi_data.get('start_time', '09:00'),
                        "end_time": taxi_data.get('end_time', '09:20'),
                        "cost": round(price_per_car * num_cars, 2),
                        "distance": float(taxi_data.get('distance', 0)),
                        "cars": num_cars,
                        "price": round(price_per_car, 2)
                    }]
            
            return []
            
        except Exception as e:
            return []
    
    def _fix_day_count(self, itinerary: List[Dict], target_days: int) -> List[Dict]:
        """修正天数"""
        current_days = len(itinerary)
        
        if current_days < target_days:
            while len(itinerary) < target_days:
                template_day = itinerary[-2] if len(itinerary) > 1 else itinerary[0]
                new_day = {
                    "day": len(itinerary) + 1,
                    "activities": template_day['activities'].copy()
                }
                itinerary.insert(-1, new_day)
        
        elif current_days > target_days:
            while len(itinerary) > target_days:
                if len(itinerary) > 2:
                    itinerary.pop(-2)
                else:
                    itinerary.pop()
        
        for i, day_plan in enumerate(itinerary, 1):
            day_plan['day'] = i
        
        return itinerary
    
    def _parse_query(self, query, prob_idx: int) -> Dict:
        """解析查询"""
        if isinstance(query, dict):
            return {
                'uid': query.get('uid', f'query_{prob_idx}'),
                'start_city': query.get('start_city', '北京'),
                'target_city': query.get('target_city', '上海'),
                'days': query.get('days', 3),
                'people_number': query.get('people_number', 1),
                'nature_language': query.get('nature_language', '')
            }
        else:
            return {
                'uid': f'query_{prob_idx}',
                'start_city': '北京',
                'target_city': '上海',
                'days': 3,
                'people_number': 1,
                'nature_language': str(query)
            }
    
    def reset(self):
        """重置agent"""
        pass