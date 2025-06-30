import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
from dataclasses import dataclass
from enum import Enum
from collections import deque
import heapq

class MarketCondition(Enum):
    BULL = "bull"          # High liquidity, favorable terms
    BEAR = "bear"          # Low liquidity, tight terms
    NEUTRAL = "neutral"    # Normal conditions
    VOLATILE = "volatile"  # Rapidly changing conditions

@dataclass
class DebtInstrument:
    instrument_id: str
    original_debt: float
    current_value: float
    risk_rating: str
    maturity_date: datetime
    debtor_profile: Dict
    payment_history: List[Dict]

@dataclass
class MarketOrder:
    order_id: str
    order_type: str  # 'buy', 'sell'
    instrument: DebtInstrument
    price: float
    quantity: float
    timestamp: datetime
    agent_id: str

class MarketMakerAgent:
    def __init__(self, agent_id: str, initial_capital: float):
        self.agent_id = agent_id
        self.capital = initial_capital
        self.inventory = {}  # Debt instruments held
        self.order_book = {"buy": [], "sell": []}
        
        # Market making parameters
        self.spread_percentage = 0.02  # 2% bid-ask spread
        self.max_position_size = initial_capital * 0.1  # 10% per position
        self.risk_limit = initial_capital * 0.3  # 30% total risk
        self.liquidity_provision_target = 0.8  # Provide liquidity 80% of time
        
        # Market analysis
        self.market_condition = MarketCondition.NEUTRAL
        self.price_history = deque(maxlen=1000)
        self.volatility_window = 100
        self.trend_window = 200
        
        # Machine learning components
        self.price_prediction_model = PricePredictionModel()
        self.risk_assessment_model = RiskAssessmentModel()
        self.liquidity_model = LiquidityOptimizationModel()
        
        # Performance tracking
        self.trades_executed = 0
        self.profit_loss = 0.0
        self.liquidity_provided = 0.0
        self.risk_events = 0
        
    def analyze_market_conditions(self) -> MarketCondition:
        """Analyze current market conditions"""
        if len(self.price_history) < self.volatility_window:
            return MarketCondition.NEUTRAL
        
        # Calculate volatility
        recent_prices = list(self.price_history)[-self.volatility_window:]
        volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        # Calculate trend
        if len(self.price_history) >= self.trend_window:
            trend_prices = list(self.price_history)[-self.trend_window:]
            trend = (trend_prices[-1] - trend_prices[0]) / trend_prices[0]
        else:
            trend = 0
        
        # Determine market condition
        if volatility > 0.15:
            return MarketCondition.VOLATILE
        elif trend > 0.1:
            return MarketCondition.BULL
        elif trend < -0.1:
            return MarketCondition.BEAR
        else:
            return MarketCondition.NEUTRAL
    
    async def evaluate_debt_instrument(self, instrument: DebtInstrument) -> Dict:
        """Comprehensive evaluation of debt instrument"""
        # Risk assessment
        risk_score = self.risk_assessment_model.assess_risk(instrument)
        
        # Price discovery
        fair_value = self._calculate_fair_value(instrument)
        market_value = self._estimate_market_value(instrument)
        
        # Liquidity assessment
        liquidity_score = self._assess_liquidity(instrument)
        
        # Recovery analysis
        recovery_probability = self._estimate_recovery_probability(instrument)
        expected_recovery = recovery_probability * instrument.original_debt
        
        return {
            "instrument_id": instrument.instrument_id,
            "risk_score": risk_score,
            "fair_value": fair_value,
            "market_value": market_value,
            "liquidity_score": liquidity_score,
            "recovery_probability": recovery_probability,
            "expected_recovery": expected_recovery,
            "recommendation": self._generate_recommendation(risk_score, fair_value, market_value)
        }
    
    def _calculate_fair_value(self, instrument: DebtInstrument) -> float:
        """Calculate fair value using DCF and risk adjustments"""
        # Base recovery rate from credit score
        credit_score = instrument.debtor_profile.get("credit_score", 600)
        base_recovery = 0.3 + (credit_score - 300) / 1000  # 30% to 80%
        
        # Adjust for payment history
        if instrument.payment_history:
            on_time_payments = sum(1 for p in instrument.payment_history if p["status"] == "on_time")
            payment_rate = on_time_payments / len(instrument.payment_history)
            base_recovery *= (0.7 + 0.3 * payment_rate)
        
        # Time value adjustment
        days_to_maturity = (instrument.maturity_date - datetime.now()).days
        discount_factor = 1 / ((1 + 0.1) ** (days_to_maturity / 365))
        
        # Risk adjustment
        risk_premium = {"AAA": 0.02, "AA": 0.04, "A": 0.06, "BBB": 0.08, "BB": 0.12, "B": 0.18, "C": 0.25}
        risk_adjustment = 1 - risk_premium.get(instrument.risk_rating, 0.20)
        
        fair_value = instrument.original_debt * base_recovery * discount_factor * risk_adjustment
        
        return fair_value
    
    def _estimate_market_value(self, instrument: DebtInstrument) -> float:
        """Estimate current market value based on recent trades"""
        # Use ML model for prediction
        features = self._extract_features(instrument)
        predicted_value = self.price_prediction_model.predict(features)
        
        # Adjust for market conditions
        market_adjustment = {
            MarketCondition.BULL: 1.05,
            MarketCondition.BEAR: 0.95,
            MarketCondition.NEUTRAL: 1.0,
            MarketCondition.VOLATILE: 0.98
        }
        
        market_value = predicted_value * market_adjustment[self.market_condition]
        
        return market_value
    
    def _assess_liquidity(self, instrument: DebtInstrument) -> float:
        """Assess liquidity of debt instrument"""
        # Factors affecting liquidity
        size_factor = min(instrument.original_debt / 100000, 1.0)  # Smaller debts more liquid
        
        rating_liquidity = {
            "AAA": 1.0, "AA": 0.9, "A": 0.8, "BBB": 0.7,
            "BB": 0.5, "B": 0.3, "C": 0.1
        }
        rating_factor = rating_liquidity.get(instrument.risk_rating, 0.2)
        
        # Time to maturity factor
        days_to_maturity = (instrument.maturity_date - datetime.now()).days
        maturity_factor = 1.0 if days_to_maturity < 90 else 0.8 if days_to_maturity < 180 else 0.6
        
        liquidity_score = (size_factor + rating_factor + maturity_factor) / 3
        
        return liquidity_score
    
    def _estimate_recovery_probability(self, instrument: DebtInstrument) -> float:
        """Estimate probability of debt recovery"""
        # Use ML model with historical data
        recovery_features = {
            "credit_score": instrument.debtor_profile.get("credit_score", 600),
            "debt_to_income": instrument.debtor_profile.get("debt_to_income", 0.5),
            "payment_history_score": self._calculate_payment_history_score(instrument.payment_history),
            "time_to_maturity": (instrument.maturity_date - datetime.now()).days,
            "debt_size": instrument.original_debt
        }
        
        probability = self.risk_assessment_model.predict_recovery(recovery_features)
        
        return probability
    
    def _calculate_payment_history_score(self, payment_history: List[Dict]) -> float:
        """Calculate payment history score"""
        if not payment_history:
            return 0.5
        
        score = 0.0
        weight_sum = 0.0
        
        for i, payment in enumerate(reversed(payment_history[-12:])):  # Last 12 payments
            weight = 1.0 / (i + 1)  # Recent payments weighted more
            
            if payment["status"] == "on_time":
                score += weight * 1.0
            elif payment["status"] == "late":
                score += weight * 0.5
            else:  # missed
                score += weight * 0.0
            
            weight_sum += weight
        
        return score / weight_sum if weight_sum > 0 else 0.5
    
    async def create_market(self, instrument: DebtInstrument) -> Dict:
        """Create market by posting bid and ask prices"""
        evaluation = await self.evaluate_debt_instrument(instrument)
        
        if evaluation["liquidity_score"] < 0.3:
            return {
                "status": "rejected",
                "reason": "Insufficient liquidity",
                "instrument_id": instrument.instrument_id
            }
        
        # Calculate bid-ask prices
        mid_price = evaluation["market_value"]
        spread = self.spread_percentage * mid_price
        
        # Adjust spread based on risk and liquidity
        spread *= (1 + (1 - evaluation["liquidity_score"]) * 0.5)
        spread *= (1 + evaluation["risk_score"] * 0.3)
        
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # Check position limits
        current_exposure = self._calculate_current_exposure()
        if current_exposure + evaluation["market_value"] > self.risk_limit:
            return {
                "status": "rejected",
                "reason": "Risk limit exceeded",
                "instrument_id": instrument.instrument_id
            }
        
        # Post orders
        bid_order = MarketOrder(
            order_id=f"bid_{datetime.now().timestamp()}",
            order_type="buy",
            instrument=instrument,
            price=bid_price,
            quantity=1.0,
            timestamp=datetime.now(),
            agent_id=self.agent_id
        )
        
        ask_order = MarketOrder(
            order_id=f"ask_{datetime.now().timestamp()}",
            order_type="sell",
            instrument=instrument,
            price=ask_price,
            quantity=1.0,
            timestamp=datetime.now(),
            agent_id=self.agent_id
        )
        
        # Add to order book
        heapq.heappush(self.order_book["buy"], (-bid_price, bid_order))
        heapq.heappush(self.order_book["sell"], (ask_price, ask_order))
        
        # Track liquidity provision
        self.liquidity_provided += evaluation["market_value"]
        
        return {
            "status": "success",
            "instrument_id": instrument.instrument_id,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "spread": spread,
            "evaluation": evaluation
        }
    
    def _calculate_current_exposure(self) -> float:
        """Calculate total current exposure"""
        exposure = 0.0
        for instrument_id, position in self.inventory.items():
            exposure += position["current_value"] * position["quantity"]
        return exposure
    
    async def execute_trade(self, incoming_order: MarketOrder) -> Optional[Dict]:
        """Execute trade against order book"""
        if incoming_order.order_type == "buy":
            # Match against sell orders
            book = self.order_book["sell"]
            if not book:
                return None
            
            best_price, best_order = heapq.heappop(book)
            
            if incoming_order.price >= best_price:
                # Execute trade
                return await self._execute_transaction(incoming_order, best_order, best_price)
        
        else:  # sell order
            # Match against buy orders
            book = self.order_book["buy"]
            if not book:
                return None
            
            neg_price, best_order = heapq.heappop(book)
            best_price = -neg_price
            
            if incoming_order.price <= best_price:
                # Execute trade
                return await self._execute_transaction(incoming_order, best_order, best_price)
        
        return None
    
    async def _execute_transaction(self, incoming_order: MarketOrder, 
                                  book_order: MarketOrder, execution_price: float) -> Dict:
        """Execute transaction between orders"""
        # Update inventory
        if incoming_order.order_type == "buy":
            # We sold to incoming buyer
            self._remove_from_inventory(book_order.instrument, book_order.quantity)
            self.capital += execution_price * book_order.quantity
        else:
            # We bought from incoming seller
            self._add_to_inventory(incoming_order.instrument, incoming_order.quantity, execution_price)
            self.capital -= execution_price * incoming_order.quantity
        
        # Record trade
        trade = {
            "trade_id": f"trade_{datetime.now().timestamp()}",
            "instrument_id": incoming_order.instrument.instrument_id,
            "buyer_id": incoming_order.agent_id if incoming_order.order_type == "buy" else self.agent_id,
            "seller_id": incoming_order.agent_id if incoming_order.order_type == "sell" else self.agent_id,
            "price": execution_price,
            "quantity": min(incoming_order.quantity, book_order.quantity),
            "timestamp": datetime.now().isoformat()
        }
        
        # Update metrics
        self.trades_executed += 1
        self.price_history.append(execution_price)
        
        # Calculate P&L
        if incoming_order.order_type == "sell":
            cost_basis = self.inventory.get(incoming_order.instrument.instrument_id, {}).get("cost_basis", 0)
            self.profit_loss += (execution_price - cost_basis) * incoming_order.quantity
        
        return trade
    
    def _add_to_inventory(self, instrument: DebtInstrument, quantity: float, price: float):
        """Add instrument to inventory"""
        if instrument.instrument_id not in self.inventory:
            self.inventory[instrument.instrument_id] = {
                "instrument": instrument,
                "quantity": 0,
                "cost_basis": 0,
                "current_value": price
            }
        
        position = self.inventory[instrument.instrument_id]
        total_cost = position["cost_basis"] * position["quantity"] + price * quantity
        position["quantity"] += quantity
        position["cost_basis"] = total_cost / position["quantity"] if position["quantity"] > 0 else 0
        position["current_value"] = price
    
    def _remove_from_inventory(self, instrument: DebtInstrument, quantity: float):
        """Remove instrument from inventory"""
        if instrument.instrument_id in self.inventory:
            position = self.inventory[instrument.instrument_id]
            position["quantity"] -= quantity
            
            if position["quantity"] <= 0:
                del self.inventory[instrument.instrument_id]
    
    async def provide_liquidity_metrics(self) -> Dict:
        """Provide liquidity metrics for the market"""
        total_bid_volume = sum(order[1].quantity * order[1].price for order in self.order_book["buy"])
        total_ask_volume = sum(order[1].quantity * order[1].price for order in self.order_book["sell"])
        
        bid_ask_spreads = []
        if self.order_book["buy"] and self.order_book["sell"]:
            best_bid = -self.order_book["buy"][0][0]
            best_ask = self.order_book["sell"][0][0]
            bid_ask_spreads.append((best_ask - best_bid) / ((best_ask + best_bid) / 2))
        
        return {
            "total_bid_volume": total_bid_volume,
            "total_ask_volume": total_ask_volume,
            "average_spread": np.mean(bid_ask_spreads) if bid_ask_spreads else 0,
            "market_depth": len(self.order_book["buy"]) + len(self.order_book["sell"]),
            "liquidity_score": self._calculate_market_liquidity_score()
        }
    
    def _calculate_market_liquidity_score(self) -> float:
        """Calculate overall market liquidity score"""
        # Factors: spread, depth, volume
        spread_score = 1.0 - min(self.spread_percentage * 10, 1.0)
        depth_score = min((len(self.order_book["buy"]) + len(self.order_book["sell"])) / 20, 1.0)
        volume_score = min(self.trades_executed / 100, 1.0)
        
        return (spread_score + depth_score + volume_score) / 3
    
    async def hedge_positions(self) -> Dict:
        """Hedge current positions to manage risk"""
        hedging_actions = []
        
        for instrument_id, position in self.inventory.items():
            risk_exposure = position["quantity"] * position["current_value"]
            
            if risk_exposure > self.max_position_size:
                # Need to reduce position
                hedge_quantity = (risk_exposure - self.max_position_size) / position["current_value"]
                
                hedge_order = MarketOrder(
                    order_id=f"hedge_{datetime.now().timestamp()}",
                    order_type="sell",
                    instrument=position["instrument"],
                    price=position["current_value"] * 0.98,  # Slight discount for quick execution
                    quantity=hedge_quantity,
                    timestamp=datetime.now(),
                    agent_id=self.agent_id
                )
                
                hedging_actions.append({
                    "instrument_id": instrument_id,
                    "action": "reduce_position",
                    "quantity": hedge_quantity,
                    "reason": "position_limit_exceeded"
                })
        
        return {
            "hedging_required": len(hedging_actions) > 0,
            "actions": hedging_actions,
            "current_exposure": self._calculate_current_exposure(),
            "risk_utilization": self._calculate_current_exposure() / self.risk_limit
        }
    
    def update_market_models(self):
        """Update ML models with recent data"""
        if len(self.price_history) >= 100:
            # Prepare training data
            features, targets = self._prepare_training_data()
            
            # Update models
            self.price_prediction_model.update(features, targets)
            self.risk_assessment_model.update_with_market_data(self.inventory, self.trades_executed)
            self.liquidity_model.optimize_parameters(self.order_book, self.price_history)
            
            # Adjust market making parameters
            self._adjust_parameters_based_on_performance()
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        prices = list(self.price_history)
        features = []
        targets = []
        
        window_size = 20
        for i in range(window_size, len(prices) - 1):
            # Features: price history, volume, volatility
            price_window = prices[i-window_size:i]
            feature_vector = [
                np.mean(price_window),
                np.std(price_window),
                (price_window[-1] - price_window[0]) / price_window[0],
                self.trades_executed,
                self.market_condition.value == "volatile"
            ]
            features.append(feature_vector)
            targets.append(prices[i + 1])
        
        return np.array(features), np.array(targets)
    
    def _adjust_parameters_based_on_performance(self):
        """Dynamically adjust market making parameters"""
        if self.profit_loss < 0:
            # Widen spreads if losing money
            self.spread_percentage = min(self.spread_percentage * 1.1, 0.05)
        elif self.profit_loss > self.capital * 0.05:
            # Tighten spreads if profitable
            self.spread_percentage = max(self.spread_percentage * 0.95, 0.01)
        
        # Adjust risk limits based on volatility
        if self.market_condition == MarketCondition.VOLATILE:
            self.max_position_size = self.capital * 0.05
        else:
            self.max_position_size = self.capital * 0.1
    
    def generate_market_report(self) -> Dict:
        """Generate comprehensive market report"""
        return {
            "agent_id": self.agent_id,
            "market_condition": self.market_condition.value,
            "capital": self.capital,
            "inventory_value": sum(p["current_value"] * p["quantity"] for p in self.inventory.values()),
            "total_assets": self.capital + sum(p["current_value"] * p["quantity"] for p in self.inventory.values()),
            "trades_executed": self.trades_executed,
            "profit_loss": self.profit_loss,
            "liquidity_provided": self.liquidity_provided,
            "current_spread": self.spread_percentage,
            "risk_utilization": self._calculate_current_exposure() / self.risk_limit,
            "market_metrics": asyncio.run(self.provide_liquidity_metrics())
        }


class PricePredictionModel:
    """ML model for price prediction"""
    
    def __init__(self):
        self.model_weights = np.random.randn(5)  # Simple linear model
        self.learning_rate = 0.01
    
    def predict(self, features: np.ndarray) -> float:
        return np.dot(features, self.model_weights)
    
    def update(self, features: np.ndarray, targets: np.ndarray):
        # Simple gradient descent
        predictions = np.dot(features, self.model_weights)
        errors = predictions - targets
        gradient = np.dot(features.T, errors) / len(features)
        self.model_weights -= self.learning_rate * gradient
    
    def _extract_features(self, instrument: DebtInstrument) -> np.ndarray:
        return np.array([
            instrument.original_debt,
            instrument.current_value,
            {"AAA": 7, "AA": 6, "A": 5, "BBB": 4, "BB": 3, "B": 2, "C": 1}.get(instrument.risk_rating, 0),
            (instrument.maturity_date - datetime.now()).days,
            len(instrument.payment_history)
        ])


class RiskAssessmentModel:
    """ML model for risk assessment"""
    
    def __init__(self):
        self.risk_weights = {
            "credit_score": 0.3,
            "debt_to_income": 0.2,
            "payment_history": 0.25,
            "time_to_maturity": 0.15,
            "debt_size": 0.1
        }
    
    def assess_risk(self, instrument: DebtInstrument) -> float:
        risk_score = 0.0
        
        # Credit score component
        credit_score = instrument.debtor_profile.get("credit_score", 600)
        risk_score += (850 - credit_score) / 550 * self.risk_weights["credit_score"]
        
        # Debt to income
        dti = instrument.debtor_profile.get("debt_to_income", 0.5)
        risk_score += min(dti, 1.0) * self.risk_weights["debt_to_income"]
        
        # Payment history
        if instrument.payment_history:
            missed_payments = sum(1 for p in instrument.payment_history if p["status"] == "missed")
            risk_score += (missed_payments / len(instrument.payment_history)) * self.risk_weights["payment_history"]
        
        # Time to maturity
        days_to_maturity = (instrument.maturity_date - datetime.now()).days
        risk_score += max(0, 1 - days_to_maturity / 365) * self.risk_weights["time_to_maturity"]
        
        # Debt size
        risk_score += min(instrument.original_debt / 100000, 1.0) * self.risk_weights["debt_size"]
        
        return min(risk_score, 1.0)
    
    def predict_recovery(self, features: Dict) -> float:
        # Simplified recovery prediction
        base_recovery = 0.5
        
        # Adjust based on features
        if features["credit_score"] > 700:
            base_recovery += 0.2
        elif features["credit_score"] < 600:
            base_recovery -= 0.2
        
        if features["payment_history_score"] > 0.8:
            base_recovery += 0.1
        elif features["payment_history_score"] < 0.3:
            base_recovery -= 0.1
        
        return max(0.1, min(0.9, base_recovery))
    
    def update_with_market_data(self, inventory: Dict, trades: int):
        # Update risk weights based on performance
        pass


class LiquidityOptimizationModel:
    """Model for optimizing liquidity provision"""
    
    def __init__(self):
        self.optimal_spread = 0.02
        self.optimal_depth = 10
    
    def optimize_parameters(self, order_book: Dict, price_history: deque):
        # Analyze order book imbalance
        buy_depth = len(order_book["buy"])
        sell_depth = len(order_book["sell"])
        
        if buy_depth > sell_depth * 1.5:
            # More buyers, can increase ask prices
            self.optimal_spread *= 1.05
        elif sell_depth > buy_depth * 1.5:
            # More sellers, can decrease bid prices
            self.optimal_spread *= 1.05
        else:
            # Balanced market, tighten spreads
            self.optimal_spread *= 0.98
        
        # Ensure reasonable bounds
        self.optimal_spread = max(0.01, min(0.05, self.optimal_spread))