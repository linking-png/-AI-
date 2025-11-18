"""
约束验证模块
用于验证生成的行程是否满足用户的各种约束条件
"""

import re
from typing import Dict, List, Any, Optional


class ConstraintValidator:
    """约束验证器"""
    
    def __init__(self):
        self.validation_results = []
    
    def validate_itinerary(self, itinerary: Dict, query: Dict) -> Dict:
        """
        验证行程是否满足所有约束
        
        Args:
            itinerary: 生成的行程
            query: 原始查询
            
        Returns:
            验证结果字典
        """
        results = {
            "valid": True,
            "violations": [],
            "warnings": [],
            "checks": {}
        }
        
        # 1. 基础约束检查
        results["checks"]["basic"] = self._validate_basic_constraints(
            itinerary, query
        )
        
        # 2. 预算约束检查
        results["checks"]["budget"] = self._validate_budget_constraints(
            itinerary, query
        )
        
        # 3. 偏好约束检查
        results["checks"]["preference"] = self._validate_preference_constraints(
            itinerary, query
        )
        
        # 4. 逻辑约束检查
        results["checks"]["logic"] = self._validate_logic_constraints(
            itinerary, query
        )
        
        # 汇总结果
        for check_type, check_result in results["checks"].items():
            if not check_result.get("valid", True):
                results["valid"] = False
                results["violations"].extend(check_result.get("violations", []))
            results["warnings"].extend(check_result.get("warnings", []))
        
        return results
    
    def _validate_basic_constraints(self, itinerary: Dict, query: Dict) -> Dict:
        """验证基础约束：人数、天数、起止城市"""
        result = {
            "valid": True,
            "violations": [],
            "warnings": []
        }
        
        # 检查天数
        expected_days = query.get('days', 3)
        actual_days = len(itinerary.get('itinerary', []))
        
        if actual_days != expected_days:
            result["valid"] = False
            result["violations"].append(
                f"天数不匹配: 期望{expected_days}天，实际{actual_days}天"
            )
        
        # 检查人数
        expected_people = query.get('people_number', 1)
        actual_people = itinerary.get('people_number', 0)
        
        if actual_people != expected_people:
            result["valid"] = False
            result["violations"].append(
                f"人数不匹配: 期望{expected_people}人，实际{actual_people}人"
            )
        
        # 检查起止城市
        expected_start = query.get('start_city')
        expected_target = query.get('target_city')
        actual_start = itinerary.get('start_city')
        actual_target = itinerary.get('target_city')
        
        if expected_start and actual_start != expected_start:
            result["violations"].append(
                f"起始城市不匹配: 期望{expected_start}，实际{actual_start}"
            )
        
        if expected_target and actual_target != expected_target:
            result["violations"].append(
                f"目标城市不匹配: 期望{expected_target}，实际{actual_target}"
            )
        
        return result
    
    def _validate_budget_constraints(self, itinerary: Dict, query: Dict) -> Dict:
        """验证预算约束"""
        result = {
            "valid": True,
            "violations": [],
            "warnings": [],
            "costs": {}
        }
        
        # 计算各项费用
        total_cost = 0
        transport_cost = 0
        accommodation_cost = 0
        dining_cost = 0
        attraction_cost = 0
        inner_transport_cost = 0
        
        for day_plan in itinerary.get('itinerary', []):
            for activity in day_plan.get('activities', []):
                activity_type = activity.get('type')
                cost = activity.get('cost', 0)
                
                total_cost += cost
                
                if activity_type in ['airplane', 'train']:
                    transport_cost += cost
                elif activity_type == 'accommodation':
                    accommodation_cost += cost
                elif activity_type in ['breakfast', 'lunch', 'dinner']:
                    dining_cost += cost
                elif activity_type == 'attraction':
                    attraction_cost += cost
                
                # 城内交通
                for trans in activity.get('transports', []):
                    trans_cost = trans.get('cost', 0)
                    inner_transport_cost += trans_cost
                    total_cost += trans_cost
        
        result["costs"] = {
            "total": total_cost,
            "transport": transport_cost,
            "accommodation": accommodation_cost,
            "dining": dining_cost,
            "attraction": attraction_cost,
            "inner_transport": inner_transport_cost
        }
        
        # 从查询中提取预算约束
        nature_language = query.get('nature_language', '')
        
        # 检查总预算
        total_budget_match = re.search(r'总预算.*?(\d+(?:\.\d+)?)', nature_language)
        if total_budget_match:
            budget = float(total_budget_match.group(1))
            if total_cost > budget:
                result["violations"].append(
                    f"总预算超支: 预算{budget}，实际{total_cost:.2f}"
                )
                result["valid"] = False
        
        # 检查交通预算
        transport_budget_match = re.search(
            r'(?:跨城市)?交通.*?预算.*?(\d+(?:\.\d+)?)', nature_language
        )
        if transport_budget_match:
            budget = float(transport_budget_match.group(1))
            if transport_cost > budget:
                result["violations"].append(
                    f"交通预算超支: 预算{budget}，实际{transport_cost:.2f}"
                )
                result["valid"] = False
        
        # 检查城内交通预算
        inner_transport_budget_match = re.search(
            r'(?:在)?城市内出行.*?预算.*?(\d+(?:\.\d+)?)', nature_language
        )
        if inner_transport_budget_match:
            budget = float(inner_transport_budget_match.group(1))
            if inner_transport_cost > budget:
                result["violations"].append(
                    f"城内交通预算超支: 预算{budget}，实际{inner_transport_cost:.2f}"
                )
                result["valid"] = False
        
        # 检查住宿预算
        accommodation_budget_match = re.search(
            r'住宿.*?预算.*?(\d+(?:\.\d+)?)', nature_language
        )
        if accommodation_budget_match:
            budget = float(accommodation_budget_match.group(1))
            if accommodation_cost > budget:
                result["violations"].append(
                    f"住宿预算超支: 预算{budget}，实际{accommodation_cost:.2f}"
                )
                result["valid"] = False
        
        # 检查用餐预算
        dining_budget_match = re.search(
            r'用餐.*?预算.*?(\d+(?:\.\d+)?)', nature_language
        )
        if dining_budget_match:
            budget = float(dining_budget_match.group(1))
            if dining_cost > budget:
                result["violations"].append(
                    f"用餐预算超支: 预算{budget}，实际{dining_cost:.2f}"
                )
                result["valid"] = False
        
        # 检查游览预算
        attraction_budget_match = re.search(
            r'游览.*?预算.*?(\d+(?:\.\d+)?)', nature_language
        )
        if attraction_budget_match:
            budget = float(attraction_budget_match.group(1))
            if attraction_cost > budget:
                result["violations"].append(
                    f"游览预算超支: 预算{budget}，实际{attraction_cost:.2f}"
                )
                result["valid"] = False
        
        return result
    
    def _validate_preference_constraints(self, itinerary: Dict, query: Dict) -> Dict:
        """验证偏好约束：房间类型、景点类型等"""
        result = {
            "valid": True,
            "violations": [],
            "warnings": []
        }
        
        nature_language = query.get('nature_language', '')
        
        # 检查房间类型
        if '单床房' in nature_language:
            expected_room_type = 1
        elif '双床房' in nature_language:
            expected_room_type = 2
        else:
            expected_room_type = None
        
        if expected_room_type:
            for day_plan in itinerary.get('itinerary', []):
                for activity in day_plan.get('activities', []):
                    if activity.get('type') == 'accommodation':
                        actual_room_type = activity.get('room_type')
                        if actual_room_type != expected_room_type:
                            result["violations"].append(
                                f"房间类型不匹配: 期望{expected_room_type}，实际{actual_room_type}"
                            )
                            result["valid"] = False
        
        # 检查免费景点
        if '只游览免费景点' in nature_language or '免费景点' in nature_language:
            for day_plan in itinerary.get('itinerary', []):
                for activity in day_plan.get('activities', []):
                    if activity.get('type') == 'attraction':
                        price = activity.get('price', 0)
                        if price > 0:
                            result["violations"].append(
                                f"包含付费景点: {activity.get('position')} (¥{price})"
                            )
                            result["valid"] = False
        
        # 检查必去景点/餐厅
        if '希望游览' in nature_language or '希望尝试' in nature_language:
            # 提取必去地点
            required_places = self._extract_required_places(nature_language)
            visited_places = self._get_visited_places(itinerary)
            
            for place in required_places:
                if place not in visited_places:
                    result["violations"].append(
                        f"缺少必去地点: {place}"
                    )
                    result["valid"] = False
        
        # 检查避免地点
        if '不希望游览' in nature_language:
            excluded_places = self._extract_excluded_places(nature_language)
            visited_places = self._get_visited_places(itinerary)
            
            for place in excluded_places:
                if place in visited_places:
                    result["violations"].append(
                        f"包含应避免的地点: {place}"
                    )
                    result["valid"] = False
        
        return result
    
    def _validate_logic_constraints(self, itinerary: Dict, query: Dict) -> Dict:
        """验证逻辑约束：时间、票数等"""
        result = {
            "valid": True,
            "violations": [],
            "warnings": []
        }
        
        people_number = itinerary.get('people_number', 1)
        
        # 检查每个活动
        for day_plan in itinerary.get('itinerary', []):
            for activity in day_plan.get('activities', []):
                activity_type = activity.get('type')
                
                # 检查票数
                if activity_type in ['attraction', 'airplane', 'train']:
                    tickets = activity.get('tickets', 0)
                    if tickets != people_number:
                        result["warnings"].append(
                            f"{activity_type} 票数({tickets})与人数({people_number})不匹配"
                        )
                
                # 检查出租车数量
                for trans in activity.get('transports', []):
                    if trans.get('mode') == 'taxi':
                        cars = trans.get('cars', 0)
                        expected_cars = (people_number + 3) // 4  # 每辆车最多4人
                        if cars != expected_cars:
                            result["warnings"].append(
                                f"出租车数量({cars})可能不合理，{people_number}人建议{expected_cars}辆"
                            )
        
        return result
    
    def _extract_required_places(self, text: str) -> List[str]:
        """从文本中提取必去地点"""
        places = []
        
        # 匹配 "希望游览XXX" 或 "希望尝试XXX"
        patterns = [
            r'希望游览[:：]?([^，。\n]+)',
            r'希望尝试[:：]?([^，。\n]+)',
            r'希望入住.*?[:：]([^，。\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 分割多个地点
                items = re.split(r'[和与]', match.strip())
                places.extend([item.strip() for item in items if item.strip()])
        
        return places
    
    def _extract_excluded_places(self, text: str) -> List[str]:
        """从文本中提取应避免的地点"""
        places = []
        
        # 匹配 "不希望游览XXX"
        pattern = r'不希望游览[:：]?([^，。\n]+)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            items = re.split(r'[和与]', match.strip())
            places.extend([item.strip() for item in items if item.strip()])
        
        return places
    
    def _get_visited_places(self, itinerary: Dict) -> List[str]:
        """获取行程中访问的所有地点"""
        places = []
        
        for day_plan in itinerary.get('itinerary', []):
            for activity in day_plan.get('activities', []):
                position = activity.get('position')
                if position:
                    places.append(position)
        
        return places


# 测试代码
if __name__ == "__main__":
    print("="*60)
    print("测试约束验证器")
    print("="*60)
    
    validator = ConstraintValidator()
    
    # 模拟行程
    test_itinerary = {
        "people_number": 2,
        "start_city": "北京",
        "target_city": "上海",
        "itinerary": [
            {
                "day": 1,
                "activities": [
                    {
                        "type": "airplane",
                        "tickets": 2,
                        "cost": 1000,
                        "transports": []
                    },
                    {
                        "type": "attraction",
                        "position": "外滩",
                        "tickets": 2,
                        "price": 0,
                        "cost": 0,
                        "transports": []
                    },
                    {
                        "type": "accommodation",
                        "position": "某酒店",
                        "room_type": 2,
                        "rooms": 1,
                        "cost": 500,
                        "transports": []
                    }
                ]
            }
        ]
    }
    
    # 模拟查询
    test_query = {
        "start_city": "北京",
        "target_city": "上海",
        "days": 1,
        "people_number": 2,
        "nature_language": "我们2人，从北京出发，到上海旅行1天，要求如下：\n在用餐上的预算为200\n希望入住双床房"
    }
    
    # 验证
    results = validator.validate_itinerary(test_itinerary, test_query)
    
    print(f"\n验证结果: {'✓ 通过' if results['valid'] else '✗ 未通过'}")
    print(f"\n违规项 ({len(results['violations'])}):")
    for violation in results['violations']:
        print(f"  - {violation}")
    
    print(f"\n警告项 ({len(results['warnings'])}):")
    for warning in results['warnings']:
        print(f"  - {warning}")
    
    print(f"\n详细检查:")
    for check_type, check_result in results['checks'].items():
        status = '✓' if check_result.get('valid', True) else '✗'
        print(f"  {status} {check_type}")
