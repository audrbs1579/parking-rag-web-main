import azure.functions as func
import logging
import json
import os
from openai import AzureOpenAI
from azure.cosmos import CosmosClient
from datetime import datetime, timezone
import pytz
import re

# 환경 변수 로드
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]
OPENAI_GPT_MODEL = os.environ["OPENAI_GPT_MODEL"]

COSMOS_ENDPOINT = os.environ["COSMOS_DB_ENDPOINT"]
COSMOS_KEY = os.environ["COSMOS_DB_KEY"]
COSMOS_DB = os.environ["COSMOS_DB_NAME"]
COSMOS_PARKING_CONTAINER = os.environ["COSMOS_DB_CONTAINER"]  # 주차장 컨테이너
COSMOS_FACILITY_CONTAINER = os.environ["COSMOS_FACILITY_CONTAINER"] # 시설 컨테이너
COSMOS_FLIGHT_CONTAINER = os.environ["COSMOS_FLIGHT_CONTAINER"]  # 항공편 컨테이너

APPLICATION_JSON = "application/json"

# 클라이언트 초기화
openai_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION
)

cosmos_client = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
db_client = cosmos_client.get_database_client(COSMOS_DB)
parking_container = db_client.get_container_client(COSMOS_PARKING_CONTAINER)
facility_container = db_client.get_container_client(COSMOS_FACILITY_CONTAINER)
flight_container = db_client.get_container_client(COSMOS_FLIGHT_CONTAINER)

# 항공편 번호 추출 함수 추가
def extract_flight_number(text):
    """텍스트에서 항공편 번호 추출"""
    # 항공편 번호 패턴: 영문 2-3자리 + 숫자 3-4자리
    flight_patterns = [
        r'\b[A-Z]{2}[0-9]{3,4}\b',  # KE123, AF5369
        r'\b[0-9][A-Z][0-9]{3,4}\b',  # 7C1301
        r'\b[A-Z][0-9][A-Z][0-9]{3,4}\b'  # 특수 패턴
    ]
    
    for pattern in flight_patterns:
        matches = re.findall(pattern, text.upper())
        if matches:
            return matches[0]
    
    return None

# 질문 분류 함수 개선
def classify_question(user_query):
    """사용자 질문을 카테고리별로 분류"""
    # 먼저 항공편 번호 추출 시도
    flight_number = extract_flight_number(user_query)
    
    messages = [
        {"role": "system", "content": """
            사용자의 질문을 다음 카테고리 중 하나로 분류하세요:
            
            1. "parking" - 주차장 관련 질문 (혼잡도, 잔여공간, 주차비 등)
            2. "facility" - 공항 시설 관련 질문 (환전소, 식당, 쇼핑, 편의시설 등)
            3. "flight" - 항공편 관련 질문 (출발/도착 시간, 게이트, 날씨, 항공사 등)
            4. "general" - 여행 준비물, 일반적인 공항 이용 팁 등
            5. "mixed" - 여러 카테고리가 섞인 복합 질문
            
            항공편 번호나 "KE1234", "7C1301", "AF5369" 같은 형태가 있으면 거의 확실히 "flight"입니다.
            "식당", "한식", "중식", "일식", "카페", "편의점", "쇼핑", "면세점", "환전" 등이 있으면 "facility"입니다.
            
            JSON 형태로만 응답하세요:
            {"category": "분류결과", "confidence": 0.95}
        """},
        {"role": "user", "content": user_query}
    ]
    
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_GPT_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        # 추출된 항공편 번호가 있으면 추가
        if flight_number:
            result["flight_number"] = flight_number
            result["category"] = "flight"  # 항공편 번호가 있으면 강제로 flight 카테고리
        else:
            result["flight_number"] = None
            
        logging.info(f"질문 분류 결과: {result}")
        return result
    except Exception as e:
        logging.error(f"질문 분류 오류: {str(e)}")
        return {"category": "general", "confidence": 0.5, "flight_number": flight_number}

# 기존 주차장 관련 함수들
def get_entities(user_query):
    kst = pytz.timezone("Asia/Seoul")
    current_datetime = datetime.now(timezone.utc).astimezone(kst)
    base_date = current_datetime.strftime("%Y%m%d")
    current_hour = current_datetime.strftime("%H")

    messages = [
        {"role": "system", "content": f"""
            당신은 주차장 질문에서 핵심 정보를 추출하는 전문가입니다.
            
            주차장 위치 키워드:
            - "T1", "터미널1", "1터미널" → "T1"
            - "T2", "터미널2", "2터미널" → "T2"  
            - "단기", "단기주차장" → "단기주차장"
            - "장기", "장기주차장" → "장기주차장"
            - "지상", "지상층" → "지상층"
            - "지하", "지하층" → "지하층"
            
            시간 정보:
            - "지금", "현재" → "{current_hour}"
            - "오전", "AM" → "0X" 형태
            - "오후", "PM" → "1X" 형태
            
            현재 날짜: {base_date}
            현재 시간: {current_hour}
            
            JSON으로만 응답하세요. 확실하지 않은 정보는 null로 설정하세요.
            
            예시:
            {{"floor_keywords": ["T1", "단기주차장", "지상층"], "date": "{base_date}", "time": "{current_hour}"}}
        """},
        {"role": "user", "content": user_query}
    ]

    res = openai_client.chat.completions.create(
        model=OPENAI_GPT_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1
    )

    return res.choices[0].message.content

def query_similar_parking_data(user_query, entities, top_k=10):
    # 임베딩 생성
    embedding_res = openai_client.embeddings.create(
        input=user_query,
        model=AZURE_OPENAI_DEPLOYMENT
    )
    query_vector = embedding_res.data[0].embedding

    # 기본 벡터 검색 (필터링 없음)
    base_query = f"""
    SELECT TOP {top_k}
        c.floor, c.parking_count, c.parking_total,
        c.congestion_rate, c.congestion_level, c.date, c.time,
        VectorDistance(c.embedding, @query_vector) as similarity_score
    FROM c
    WHERE IS_DEFINED(c.embedding)
    ORDER BY VectorDistance(c.embedding, @query_vector)
    """
    
    parameters = [{"name": "@query_vector", "value": query_vector}]
    
    try:
        results = list(parking_container.query_items(
            query=base_query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        logging.info(f"벡터 검색 결과 수: {len(results)}")
        
        # 엔티티 기반 후처리 필터링
        if entities and results:
            filtered_results = []
            floor_keywords = entities.get("floor_keywords", [])
            target_date = entities.get("date")
            target_time = entities.get("time")
            
            for result in results:
                score = 0
                
                # 위치 키워드 매칭 (부분 일치)
                if floor_keywords:
                    floor_text = result.get("floor", "").upper()
                    for keyword in floor_keywords:
                        if keyword.upper() in floor_text:
                            score += 2
                
                # 날짜 매칭
                if target_date and str(result.get("date", "")) == str(target_date):
                    score += 1
                
                # 시간 매칭 (1시간 범위)
                if target_time and result.get("time"):
                    result_hour = int(result["time"].split(":")[0])
                    target_hour = int(target_time)
                    if abs(result_hour - target_hour) <= 1:
                        score += 1
                
                result["relevance_score"] = score
                filtered_results.append(result)
            
            # 관련성 점수로 정렬
            filtered_results.sort(key=lambda x: (x["relevance_score"], -x["similarity_score"]), reverse=True)
            
            return filtered_results[:top_k//2] if filtered_results else results[:top_k//2]
        
        return results
        
    except Exception as e:
        logging.error(f"벡터 검색 오류: {str(e)}")
        return fallback_search(entities)

def fallback_search(entities):
    try:
        where_clauses = []
        parameters = []
        
        if entities and entities.get("floor_keywords"):
            floor_conditions = []
            for i, keyword in enumerate(entities["floor_keywords"]):
                param_name = f"@floor_keyword_{i}"
                floor_conditions.append(f"CONTAINS(UPPER(c.floor), UPPER({param_name}))")
                parameters.append({"name": param_name, "value": keyword})
            
            if floor_conditions:
                where_clauses.append(f"({' OR '.join(floor_conditions)})")
        
        if entities and entities.get("date"):
            where_clauses.append("c.date = @date")
            parameters.append({"name": "@date", "value": int(entities["date"])})
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        query = f"""
        SELECT TOP 5
            c.floor, c.parking_count, c.parking_total,
            c.congestion_rate, c.congestion_level, c.date, c.time
        FROM c
        WHERE {where_sql}
        ORDER BY c.date DESC, c.time DESC
        """
        
        results = list(parking_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        logging.info(f"Fallback 검색 결과 수: {len(results)}")
        return results
        
    except Exception as e:
        logging.error(f"Fallback 검색 오류: {str(e)}")
        return []

# 시설 관련 함수들 개선
def query_facility_data(user_query, top_k=10):
    """공항 시설 정보 검색 - 더 많은 결과 반환"""
    try:
        # 임베딩 생성
        embedding_res = openai_client.embeddings.create(
            input=user_query,
            model=AZURE_OPENAI_DEPLOYMENT
        )
        query_vector = embedding_res.data[0].embedding
        
        # 벡터 검색
        query = f"""
        SELECT TOP {top_k}
            c.entrpskoreannm, c.trtmntprdlstkoreannm, c.lckoreannm,
            c.servicetime, c.arrordep, c.tel,
            VectorDistance(c.embedding, @query_vector) as similarity_score
        FROM c
        WHERE IS_DEFINED(c.embedding)
        ORDER BY VectorDistance(c.embedding, @query_vector)
        """
        
        parameters = [{"name": "@query_vector", "value": query_vector}]
        
        results = list(facility_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True
        ))
        
        logging.info(f"시설 검색 결과 수: {len(results)}")
        
        # 키워드 기반 후처리 필터링 추가
        if results and user_query:
            filtered_results = []
            query_keywords = user_query.lower().split()
            
            for result in results:
                score = result.get('similarity_score', 0)
                
                # 키워드 매칭으로 점수 보정
                facility_text = f"{result.get('entrpskoreannm', '')} {result.get('trtmntprdlstkoreannm', '')} {result.get('lckoreannm', '')}".lower()
                
                for keyword in query_keywords:
                    if keyword in facility_text:
                        score += 0.1  # 키워드 매칭 보너스
                
                result['adjusted_score'] = score
                filtered_results.append(result)
            
            # 조정된 점수로 정렬
            filtered_results.sort(key=lambda x: x['adjusted_score'])
            return filtered_results
        
        return results
        
    except Exception as e:
        logging.error(f"시설 검색 오류: {str(e)}")
        return []

# 항공편 관련 함수들 대폭 개선
def query_flight_data(user_query, flight_number=None, top_k=10):
    """항공편 정보 검색 - 개선된 버전"""
    try:
        # 항공편 번호가 있으면 우선 정확한 매칭 시도
        if flight_number:
            logging.info(f"항공편 번호로 검색 시도: {flight_number}")
            
            # 정확한 매칭 시도
            exact_query = """
            SELECT TOP 20
                c.date, c.hr, c.min, c.yoil, c.airline, c.flightid,
                c.scheduleDateTime, c.estimatedDateTime, c.airport, c.remark,
                c.gatenumber, c.temp, c.senstemp, c.himidity, c.wind
            FROM c
            WHERE UPPER(c.flightid) = UPPER(@flight_number)
            ORDER BY c.date DESC, c.hr DESC, c.min DESC
            """
            
            exact_params = [{"name": "@flight_number", "value": flight_number}]
            
            try:
                exact_results = list(flight_container.query_items(
                    query=exact_query,
                    parameters=exact_params,
                    enable_cross_partition_query=True
                ))
                
                logging.info(f"정확한 항공편 매칭 결과 수: {len(exact_results)}")
                
                if exact_results:
                    return exact_results
                else:
                    # 부분 매칭 시도
                    partial_query = """
                    SELECT TOP 20
                        c.date, c.hr, c.min, c.yoil, c.airline, c.flightid,
                        c.scheduleDateTime, c.estimatedDateTime, c.airport, c.remark,
                        c.gatenumber, c.temp, c.senstemp, c.himidity, c.wind
                    FROM c
                    WHERE CONTAINS(UPPER(c.flightid), UPPER(@flight_number))
                    ORDER BY c.date DESC, c.hr DESC, c.min DESC
                    """
                    
                    partial_results = list(flight_container.query_items(
                        query=partial_query,
                        parameters=exact_params,
                        enable_cross_partition_query=True
                    ))
                    
                    logging.info(f"부분 항공편 매칭 결과 수: {len(partial_results)}")
                    
                    if partial_results:
                        return partial_results
                    
            except Exception as e:
                logging.error(f"항공편 직접 검색 오류: {str(e)}")
        
        # 벡터 검색 시도
        logging.info("항공편 벡터 검색 시도")
        
        embedding_res = openai_client.embeddings.create(
            input=user_query,
            model=AZURE_OPENAI_DEPLOYMENT
        )
        query_vector = embedding_res.data[0].embedding
        
        vector_query = f"""
        SELECT TOP {top_k}
            c.date, c.hr, c.min, c.yoil, c.airline, c.flightid,
            c.scheduleDateTime, c.estimatedDateTime, c.airport, c.remark,
            c.gatenumber, c.temp, c.senstemp, c.himidity, c.wind,
            VectorDistance(c.embedding, @query_vector) as similarity_score
        FROM c
        WHERE IS_DEFINED(c.embedding)
        ORDER BY VectorDistance(c.embedding, @query_vector)
        """
        
        vector_params = [{"name": "@query_vector", "value": query_vector}]
        
        results = list(flight_container.query_items(
            query=vector_query,
            parameters=vector_params,
            enable_cross_partition_query=True
        ))
        
        logging.info(f"항공편 벡터 검색 결과 수: {len(results)}")
        return results
        
    except Exception as e:
        logging.error(f"항공편 검색 전체 오류: {str(e)}")
        return []

# 통합 응답 생성 함수 개선
def generate_comprehensive_response(user_query, category, flight_number=None):
    """카테고리에 따른 통합 응답 생성"""
    try:
        context_parts = []
        
        # 주차장 정보가 필요한 경우
        if category in ["parking", "mixed"]:
            entities_str = get_entities(user_query)
            try:
                entities = json.loads(entities_str)
            except:
                entities = {}
            
            parking_data = query_similar_parking_data(user_query, entities)
            if parking_data:
                context_parts.append("🚗 주차장 현황:")
                for i, item in enumerate(parking_data[:3], 1):
                    available = item['parking_total'] - item['parking_count']
                    context_parts.append(
                        f"{i}. {item['floor']} - "
                        f"혼잡도: {item['congestion_level']}({item['congestion_rate']}%), "
                        f"잔여: {available}대"
                    )
        
        # 시설 정보가 필요한 경우 - 더 많은 결과 표시
        if category in ["facility", "mixed", "general"]:
            facility_data = query_facility_data(user_query, top_k=15)
            if facility_data:
                context_parts.append("\n🏢 공항 시설 정보:")
                # 최대 8개까지 표시 (기존 3개에서 증가)
                for i, item in enumerate(facility_data[:8], 1):
                    location = item.get('lckoreannm', '')
                    service_time = item.get('servicetime', '')
                    tel = item.get('tel', '')
                    arrordep = item.get('arrordep', '')
                    
                    context_parts.append(
                        f"{i}. {item.get('entrpskoreannm', '')}\n"
                        f"   • 서비스: {item.get('trtmntprdlstkoreannm', '')}\n"
                        f"   • 위치: {location}\n"
                        f"   • 구분: {arrordep}\n"
                        f"   • 운영시간: {service_time}\n"
                        f"   • 연락처: {tel}"
                    )
        
        # 항공편 정보가 필요한 경우
        if category in ["flight", "mixed"] or flight_number:
            flight_data = query_flight_data(user_query, flight_number)
            if flight_data:
                context_parts.append("\n✈️ 항공편 정보:")
                for i, item in enumerate(flight_data[:5], 1):
                    weather_info = ""
                    if item.get('temp'):
                        weather_info = f"날씨: {item['temp']}°C (체감 {item.get('senstemp', 'N/A')}°C), 습도: {item.get('himidity', 'N/A')}%"
                    
                    context_parts.append(
                        f"{i}. {item.get('airline', '')} {item.get('flightid', '')}\n"
                        f"   • 목적지: {item.get('airport', '')}\n"
                        f"   • 날짜: {item.get('date', '')} ({item.get('yoil', '')})\n"
                        f"   • 예정시간: {item.get('scheduleDateTime', '')}\n"
                        f"   • 예상시간: {item.get('estimatedDateTime', '')}\n"
                        f"   • 게이트: {item.get('gatenumber', '')}\n"
                        f"   • 구분: {item.get('remark', '')}\n"
                        f"   • {weather_info}"
                    )
            else:
                context_parts.append("\n✈️ 항공편 정보를 찾을 수 없습니다.")
                if flight_number:
                    context_parts.append(f"검색한 항공편 번호: {flight_number}")
        
        # 컨텍스트 조합
        context = "\n".join(context_parts) if context_parts else "관련 정보를 찾을 수 없습니다."
        
        # GPT 응답 생성
        system_prompt = """
        당신은 인천국제공항의 종합 안내 챗봇입니다.
        
        역할:
        1. 주차장 정보 (혼잡도, 잔여공간 등) 안내
        2. 공항 시설 (환전소, 식당, 편의시설 등) 안내 - 가능한 많은 옵션 제공
        3. 항공편 정보 (출발/도착 시간, 게이트, 날씨 등) 안내
        4. 여행 준비물 및 공항 이용 팁 제공
        
        응답 가이드라인:
        - 친절하고 도움이 되는 톤으로 답변
        - 구체적이고 실용적인 정보 제공
        - 이모지를 적절히 사용하여 가독성 향상
        - 사용자의 구체적인 상황을 고려한 맞춤형 답변
        - 시설 정보 요청시 가능한 많은 옵션을 제공하여 선택의 폭을 넓혀주세요
        - 항공편 정보를 찾을 수 없는 경우 명확히 안내해주세요
        - 추가적인 팁이나 주의사항이 있으면 함께 제공
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"질문: {user_query}\n\n관련 정보:\n{context}"}
        ]
        
        response = openai_client.chat.completions.create(
            model=OPENAI_GPT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500  # 토큰 수 증가
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"통합 응답 생성 오류: {str(e)}")
        return f"죄송합니다. 처리 중 오류가 발생했습니다: {str(e)}"

# 기존 주차장 전용 함수 (하위 호환성 유지)
def generate_parking_response(user_query):
    """주차장 전용 응답 생성 (기존 호환성 유지)"""
    return generate_comprehensive_response(user_query, "parking")

# Static Web App용 메인 함수
async def main(req: func.HttpRequest) -> func.HttpResponse:
    """Static Web App에서 호출되는 메인 함수"""
    
    # CORS 헤더 설정
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization"
    }
    
    try:
        # OPTIONS 요청 처리 (CORS preflight)
        if req.method == "OPTIONS":
            return func.HttpResponse(
                "",
                headers=headers,
                status_code=200
            )
        
        # POST 요청만 처리
        if req.method != "POST":
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "POST 요청만 지원됩니다"}),
                mimetype=APPLICATION_JSON,
                headers=headers,
                status_code=405
            )
        
        # 요청 본문 파싱
        body = req.get_json()
        if not body or "question" not in body:
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "질문을 입력해주세요"}),
                mimetype=APPLICATION_JSON,
                headers=headers,
                status_code=400
            )
        
        question = body.get("question", "").strip()
        if not question:
            return func.HttpResponse(
                json.dumps({"status": "error", "message": "질문이 비어있습니다"}),
                mimetype=APPLICATION_JSON,
                headers=headers,
                status_code=400
            )
        
        logging.info(f"질문 수신: {question}")
        
        # 질문 분류
        classification = classify_question(question)
        category = classification.get("category", "general")
        flight_number = classification.get("flight_number")
        
        logging.info(f"분류된 카테고리: {category}, 항공편 번호: {flight_number}")
        
        # 통합 응답 생성
        answer = generate_comprehensive_response(question, category, flight_number)
        
        return func.HttpResponse(
            json.dumps({
                "status": "success", 
                "answer": answer,
                "category": category,
                "flight_number": flight_number,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }, ensure_ascii=False),
            mimetype=APPLICATION_JSON,
            headers=headers,
            status_code=200
        )

    except Exception as e:
        logging.error(f"전체 처리 오류: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "status": "error", 
                "message": "서버 오류가 발생했습니다",
                "error_detail": str(e)
            }),
            mimetype=APPLICATION_JSON,
            headers=headers,
            status_code=500
        )
