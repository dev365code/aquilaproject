import copy
import time
import numpy as np
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from collections import deque
from .strategy import Strategy
from ..log_manager import LogManager
from ..date_converter import DateConverter


class StrategyWnt(Strategy):
    """
    워뇨띠 매매법 기반 AI 자동화 전략
    
    Wonyotti Trading Method Based AI Automation Strategy
    
    핵심 원칙:
    1. 프랙탈 기반 패턴 분석 ("역사는 되풀이된다")
    2. 캔들과 거래량 병행 분석 (거래량 = 신뢰도 지표)
    3. 철저한 리스크 관리 (시드 20% 격리, 풀시드 4배 이하)
    4. 단타 중심 매매 (하루 20~30번, 10분 내 결정)
    5. 패턴 이탈 시 즉시 손절, 패턴 완료 시 익절
    """
    
    CODE = "WNT"
    NAME = "Wonyotti-AI-Strategy"
    
    # 워뇨띠 핵심 설정값들 (피드백 반영)
    COMMISSION_RATIO = 0.0005
    PATTERN_WINDOW = 20                    # 20봉 패턴 분석
    PATTERN_SIMILARITY_THRESHOLD = 0.85    # 85% 이상 유사도
    VOLUME_CONFIDENCE_MULTIPLIER = 1.5     # 평균 거래량의 1.5배 이상
    MAX_HOLD_MINUTES = 10                  # 10분 내 결정 (워뇨띠 단타 스타일)
    
    # 리스크 관리 (워뇨띠 원칙)
    ISOLATED_MARGIN_RATIO = 0.2            # 시드의 20%만 사용
    MAX_LEVERAGE = 10                      # 격리 10배 (실질 풀시드 2배)
    PROFIT_TARGET_MIN = 0.01               # 최소 1% 익절
    PROFIT_TARGET_MAX = 0.04               # 최대 4% 익절
    STOP_LOSS_RATIO = 0.01                 # 1% 손절 (패턴 틀어질 때)
    HARD_STOP_LOSS = 0.05                  # 5% 강제 손절 (멘탈 한계)
    WITHDRAWAL_RATIO = 0.25                # 수익의 25% 출금
    
    def __init__(self):
        self.is_intialized = False
        self.is_simulation = False
        
        # 기본 상태 관리 (Strategy 인터페이스 준수)
        self.data = []
        self.budget = 0
        self.balance = 0
        self.asset_amount = 0
        self.asset_price = 0
        self.min_price = 0
        self.result = []
        self.request = None
        
        # 워뇨띠 전용 상태 관리
        self.position_status = "EMPTY"      # EMPTY/BUYING/HOLDING/SELLING
        self.entry_price = 0
        self.entry_time = None
        self.entry_pattern = None
        self.expected_exit_pattern = None
        
        # 패턴 분석용 데이터 버퍼
        self.candle_buffer = deque(maxlen=300)  # 최대 300봉 저장
        self.volume_buffer = deque(maxlen=300)
        
        # 워뇨띠 핵심 모듈들 (나중에 구현)
        self.pattern_matcher = None         # 패턴 매칭 엔진
        self.volume_analyzer = None         # 거래량 분석기
        self.risk_manager = None            # 리스크 관리자
        
        self.logger = LogManager.get_logger(__class__.__name__)
        self.waiting_requests = {}
        
        # 통계 추적
        self.daily_trade_count = 0
        self.total_profit = 0
        self.win_count = 0
        self.trade_count = 0
        
    def initialize(
        self,
        budget,
        min_price=5000,
        add_spot_callback=None,
        add_line_callback=None,
        alert_callback=None,
    ):
        """전략 초기화"""
        if self.is_intialized:
            return
            
        self.is_intialized = True
        self.budget = budget
        self.balance = budget
        self.min_price = min_price
        
        # 워뇨띠 모듈들 초기화 (나중에 구현)
        # self.pattern_matcher = WonyottiPatternMatcher()
        # self.volume_analyzer = WonyottiVolumeAnalyzer() 
        # self.risk_manager = WonyottiRiskManager(budget)
        
        self.logger.info(f"WNT Strategy initialized with budget: {budget}")
        
    def update_trading_info(self, info):
        """
        새로운 거래 정보 업데이트 - 워뇨띠 방식으로 분석
        
        워뇨띠 핵심 체크리스트:
        1. 캔들 패턴 저장 및 분석
        2. 거래량 신뢰도 체크
        3. 프랙탈 패턴 매칭
        4. 현재 포지션 상태에 따른 분석
        """
        if not self.is_intialized:
            return
            
        # 기본 데이터 처리
        target = None
        for item in info:
            if item["type"] == "primary_candle":
                target = item
                break
                
        if target is None:
            return
            
        self.data.append(copy.deepcopy(target))
        
        # 워뇨띠 분석 프로세스
        self._update_wonyotti_analysis(target)
        
    def _update_wonyotti_analysis(self, candle_data):
        """워뇨띠 방식의 시장 분석"""
        try:
            current_price = candle_data["closing_price"]
            current_volume = candle_data.get("acc_volume", 0)
            current_time = candle_data["date_time"]
            
            # 캔들과 거래량 데이터 저장
            self.candle_buffer.append({
                'open': candle_data.get("opening_price", current_price),
                'high': candle_data.get("high_price", current_price),
                'low': candle_data.get("low_price", current_price),
                'close': current_price,
                'volume': current_volume,
                'time': current_time
            })
            self.volume_buffer.append(current_volume)
            
            # 최소 데이터 확보 후 분석 시작
            if len(self.candle_buffer) < self.PATTERN_WINDOW:
                return
                
            # 1. 워뇨띠 거래량 신뢰도 체크
            if not self._check_volume_confidence():
                self.logger.debug("[WNT] Volume confidence too low, skipping analysis")
                return
                
            # 2. 현재 포지션 상태에 따른 분석
            if self.position_status == "EMPTY":
                self._analyze_entry_opportunity(candle_data)
            elif self.position_status == "HOLDING":
                self._analyze_exit_opportunity(candle_data)
                
        except (KeyError, TypeError, ValueError) as err:
            self.logger.error(f"[WNT] Analysis error: {err}")
            
    def _check_volume_confidence(self):
        """워뇨띠 원칙: 거래량 신뢰도 체크"""
        if len(self.volume_buffer) < 7:
            return False
            
        current_volume = self.volume_buffer[-1]
        avg_volume = np.mean(list(self.volume_buffer)[-7:])  # 최근 7개 평균
        
        # 워뇨띠: "거래량이 신뢰할 만한 수준에 이르기 전에는 매매 안함"
        volume_confidence = current_volume >= (avg_volume * self.VOLUME_CONFIDENCE_MULTIPLIER)
        
        if not volume_confidence:
            self.logger.debug(f"[WNT] Volume too low: {current_volume} < {avg_volume * self.VOLUME_CONFIDENCE_MULTIPLIER}")
            
        return volume_confidence
        
    def _analyze_entry_opportunity(self, candle_data):
        """매수 진입 기회 분석 - 워뇨띠 방식"""
        # TODO: 프랙탈 패턴 매칭 구현
        # 현재는 기본 로직으로 대체
        
        current_pattern = list(self.candle_buffer)[-self.PATTERN_WINDOW:]
        
        # 워뇨띠 패턴 체크 (임시 - 나중에 AI로 대체)
        if self._is_wonyotti_buy_pattern(current_pattern):
            self.position_status = "BUYING"
            self.entry_time = datetime.now()
            
            self.logger.info("[WNT] Entry pattern detected - preparing to buy")
        
    def _analyze_exit_opportunity(self, candle_data):
        """매도 청산 기회 분석 - 워뇨띠 방식"""
        if not self.entry_price or not self.entry_time:
            return
            
        current_price = candle_data["closing_price"]
        profit_ratio = (current_price - self.entry_price) / self.entry_price
        hold_time = (datetime.now() - self.entry_time).total_seconds() / 60  # 분 단위
        
        # 워뇨띠 청산 조건들
        should_exit = False
        exit_reason = ""
        
        # 1. 패턴 이탈 시 즉시 손절 (워뇨띠 핵심)
        if self._is_pattern_broken():
            should_exit = True
            exit_reason = "Pattern broken - immediate stop loss"
            
        # 2. 수익 목표 달성 시 익절
        elif profit_ratio >= self.PROFIT_TARGET_MIN:
            should_exit = True
            exit_reason = f"Profit target reached: {profit_ratio:.2%}"
            
        # 3. 손절선 도달
        elif profit_ratio <= -self.STOP_LOSS_RATIO:
            should_exit = True
            exit_reason = f"Stop loss triggered: {profit_ratio:.2%}"
            
        # 4. 강제 손절 (멘탈 한계)
        elif profit_ratio <= -self.HARD_STOP_LOSS:
            should_exit = True
            exit_reason = f"Hard stop loss: {profit_ratio:.2%}"
            
        # 5. 시간 초과 (워뇨띠: 10분 내 결정)
        elif hold_time > self.MAX_HOLD_MINUTES:
            should_exit = True
            exit_reason = f"Time limit exceeded: {hold_time:.1f}min"
            
        if should_exit:
            self.position_status = "SELLING" 
            self.logger.info(f"[WNT] Exit signal: {exit_reason}")
    
    def _is_wonyotti_buy_pattern(self, pattern):
        """워뇨띠 매수 패턴 감지 (임시 구현)"""
        # TODO: 실제 프랙탈 패턴 매칭으로 교체
        if len(pattern) < 3:
            return False
            
        # 임시: 단순한 상승 패턴 체크
        recent_closes = [candle['close'] for candle in pattern[-3:]]
        is_uptrend = recent_closes[-1] > recent_closes[-2] > recent_closes[-3]
        
        # 거래량 증가 체크
        recent_volumes = [candle['volume'] for candle in pattern[-3:]]
        is_volume_increasing = recent_volumes[-1] > recent_volumes[-2]
        
        return is_uptrend and is_volume_increasing
        
    def _is_pattern_broken(self):
        """패턴 이탈 감지 (임시 구현)"""
        # TODO: 실제 프랙탈 패턴 비교로 교체
        if len(self.candle_buffer) < 5:
            return False
            
        # 임시: 급격한 하락 감지
        recent_closes = [candle['close'] for candle in list(self.candle_buffer)[-5:]]
        price_drop = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        
        return price_drop < -0.02  # 2% 급락 시 패턴 이탈로 판단
        
    def get_request(self):
        """
        워뇨띠 방식 매매 요청 생성
        
        워뇨띠 원칙:
        - 확실한 패턴일 때만 진입
        - 빠른 결정과 실행
        - 분할 매매로 리스크 분산
        """
        if not self.is_intialized or not self.data:
            return None
            
        try:
            current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            if self.is_simulation and self.data:
                current_time = self.data[-1]["date_time"]
                
            # 워뇨띠 매매 상태별 처리
            if self.position_status == "BUYING":
                return self._create_wonyotti_buy_request(current_time)
            elif self.position_status == "SELLING":
                return self._create_wonyotti_sell_request(current_time)
                
            return None
            
        except (ValueError, KeyError, IndexError) as err:
            self.logger.error(f"[WNT] Request generation error: {err}")
            return None
            
    def _create_wonyotti_buy_request(self, current_time):
        """워뇨띠 방식 매수 요청"""
        if not self.data:
            return None
            
        current_price = float(self.data[-1]["closing_price"])
        
        # 워뇨띠 자금 관리: 시드의 20%만 사용
        available_budget = self.balance * self.ISOLATED_MARGIN_RATIO
        
        # 분할 매수 (3회 분할)
        order_budget = available_budget / 3
        
        if order_budget < self.min_price:
            self.logger.warning(f"[WNT] Order budget too small: {order_budget}")
            return None
            
        # 수수료 고려
        order_budget -= order_budget * self.COMMISSION_RATIO
        amount = order_budget / current_price
        
        # 소숫점 처리
        amount = Decimal(str(amount)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        
        if amount <= 0:
            return None
            
        request = {
            "id": DateConverter.timestamp_id(),
            "type": "buy",
            "price": str(current_price),
            "amount": str(amount.normalize()),
            "date_time": current_time,
        }
        
        self.logger.info(f"[WNT] Buy request: {amount} at {current_price}")
        return [request]
        
    def _create_wonyotti_sell_request(self, current_time):
        """워뇨띠 방식 매도 요청"""
        if self.asset_amount <= 0:
            return None
            
        current_price = float(self.data[-1]["closing_price"])
        
        # 전량 매도 (워뇨띠: 패턴 틀어지면 즉시 청산)
        amount = self.asset_amount
        
        amount = Decimal(str(amount)).quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        
        if amount <= 0:
            return None
            
        request = {
            "id": DateConverter.timestamp_id(), 
            "type": "sell",
            "price": str(current_price),
            "amount": str(amount.normalize()),
            "date_time": current_time,
        }
        
        self.logger.info(f"[WNT] Sell request: {amount} at {current_price}")
        return [request]
        
    def update_result(self, result):
        """
        거래 결과 업데이트 - 워뇨띠 학습 방식
        
        워뇨띠 방식:
        - 매 거래 후 패턴 성공률 업데이트
        - 수익 발생 시 일부 출금
        - 통계 추적으로 전략 개선
        """
        if not self.is_intialized:
            return
            
        try:
            request = result["request"]
            
            # 대기 중인 요청 관리
            if result["state"] == "requested":
                self.waiting_requests[request["id"]] = result
                return
            elif result["state"] == "done" and request["id"] in self.waiting_requests:
                del self.waiting_requests[request["id"]]
                
            # 거래 성공 시 처리
            if result["msg"] == "success":
                self._process_successful_trade(result)
                
            # 결과 저장
            self.result.append(copy.deepcopy(result))
            
        except (AttributeError, TypeError, KeyError) as err:
            self.logger.error(f"[WNT] Result processing error: {err}")
            
    def _process_successful_trade(self, result):
        """성공한 거래 처리 - 워뇨띠 방식"""
        trade_type = result["type"]
        price = float(result["price"])
        amount = float(result["amount"])
        total = price * amount
        fee = total * self.COMMISSION_RATIO
        
        if trade_type == "buy":
            # 매수 완료
            self.balance -= round(total + fee)
            self.asset_amount = round(self.asset_amount + amount, 6)
            self.asset_price = price  # 평균 단가 계산 생략 (단순화)
            
            self.position_status = "HOLDING"
            self.entry_price = price
            self.entry_time = datetime.now()
            
            self.logger.info(f"[WNT] Buy completed: {amount} at {price}")
            
        elif trade_type == "sell":
            # 매도 완료
            self.balance += round(total - fee)
            self.asset_amount = round(self.asset_amount - amount, 6)
            
            # 수익 계산 및 통계 업데이트
            if self.entry_price:
                profit_ratio = (price - self.entry_price) / self.entry_price
                profit_amount = (price - self.entry_price) * amount
                
                self.total_profit += profit_amount
                self.trade_count += 1
                
                if profit_ratio > 0:
                    self.win_count += 1
                    
                    # 워뇨띠 원칙: 수익 시 25% 출금
                    if profit_amount > 0:
                        withdrawal = profit_amount * self.WITHDRAWAL_RATIO
                        self.balance -= withdrawal
                        self.logger.info(f"[WNT] Withdrawn: {withdrawal:.0f} (25% of profit)")
                
                win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0
                self.logger.info(f"[WNT] Trade completed: {profit_ratio:.2%} profit, Win rate: {win_rate:.1%}")
            
            self.position_status = "EMPTY"
            self.entry_price = 0
            self.entry_time = None
            self.daily_trade_count += 1
            
        # 잔고 및 자산 로깅
        total_value = self.balance + (self.asset_amount * price if self.asset_amount > 0 else 0)
        self.logger.info(f"[WNT] Balance: {self.balance:.0f}, Asset: {self.asset_amount:.4f}, Total: {total_value:.0f}")