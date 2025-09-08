import asyncio
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Callable
import schedule

from core.config import AppConfig
from .processor import BatchProcessor

logger = logging.getLogger(__name__)

class BatchScheduler:
    """Manages scheduled batch processing of financial data"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.processor = BatchProcessor(config)
        
        self._running = False
        self._scheduler_thread = None
        self._stop_event = threading.Event()
        
        # Scheduled jobs
        self._jobs = {}
        
        # Statistics
        self._stats = {
            'total_jobs_run': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'last_run': None,
            'next_run': None,
            'uptime_start': datetime.now(timezone.utc)
        }
        
        # Initialize default schedules
        self._setup_default_schedules()
        
        logger.info("Batch Scheduler initialized")
    
    def _setup_default_schedules(self):
        """Setup default scheduled jobs"""
        try:
            # High-frequency data fetching (every 5 minutes during market hours for professional-grade live updates)
            self.add_job(
                name="high_frequency_fetch",
                func=self._run_high_frequency_fetch,
                schedule_type="interval",
                interval_minutes=5,  # Increased frequency for professional live monitoring
                enabled=True
            )
            
            # Real-time breaking news monitor (every 2 minutes)
            self.add_job(
                name="breaking_news_monitor",
                func=self._run_breaking_news_scan,
                schedule_type="interval",
                interval_minutes=2,
                enabled=True
            )
            
            # Pre-market catalyst scan (5:30 AM UTC - catches overnight news)
            self.add_job(
                name="pre_market_catalyst_scan",
                func=self._run_pre_market_scan,
                schedule_type="daily",
                time="05:30",
                enabled=True
            )
            
            # Daily comprehensive scan (once per day at 6 AM UTC)
            self.add_job(
                name="daily_comprehensive_scan",
                func=self._run_daily_comprehensive_scan,
                schedule_type="daily",
                time="06:00",
                enabled=True
            )
            
            # Hourly trending analysis (every hour)
            self.add_job(
                name="hourly_trending_analysis",
                func=self._run_trending_analysis,
                schedule_type="interval",
                interval_minutes=60,
                enabled=True
            )
            
            # Market hours intensive monitoring (every 1 minute during market hours)
            self.add_job(
                name="market_hours_intensive_scan",
                func=self._run_market_hours_intensive_scan,
                schedule_type="interval",
                interval_minutes=1,
                enabled=True
            )
            
            # Earnings announcement real-time tracker (every 30 seconds during earnings season)
            self.add_job(
                name="earnings_realtime_tracker",
                func=self._run_earnings_realtime_scan,
                schedule_type="interval",
                interval_minutes=0.5,  # 30 seconds
                enabled=True
            )
            
            # Weekly cleanup (every Sunday at 2 AM UTC)
            self.add_job(
                name="weekly_cleanup",
                func=self._run_weekly_cleanup,
                schedule_type="weekly",
                day="sunday",
                time="02:00",
                enabled=True
            )
            
            logger.info(f"Setup {len(self._jobs)} default scheduled jobs")
            
        except Exception as e:
            logger.error(f"Error setting up default schedules: {e}")
    
    def add_job(self, name: str, func: Callable, schedule_type: str, 
                enabled: bool = True, **schedule_kwargs) -> bool:
        """Add a new scheduled job"""
        try:
            job_config = {
                'name': name,
                'func': func,
                'schedule_type': schedule_type,
                'enabled': enabled,
                'last_run': None,
                'next_run': None,
                'run_count': 0,
                'success_count': 0,
                'failure_count': 0,
                **schedule_kwargs
            }
            
            self._jobs[name] = job_config
            
            # Schedule the job
            if enabled:
                self._schedule_job(job_config)
            
            logger.info(f"Added job: {name} ({schedule_type})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding job {name}: {e}")
            return False
    
    def _schedule_job(self, job_config: Dict[str, Any]):
        """Schedule a single job with the schedule library"""
        try:
            name = job_config['name']
            func = job_config['func']
            schedule_type = job_config['schedule_type']
            
            if schedule_type == "interval":
                if 'interval_minutes' in job_config:
                    schedule.every(job_config['interval_minutes']).minutes.do(
                        self._execute_job, name, func
                    ).tag(name)
                elif 'interval_hours' in job_config:
                    schedule.every(job_config['interval_hours']).hours.do(
                        self._execute_job, name, func
                    ).tag(name)
            
            elif schedule_type == "daily":
                time_str = job_config.get('time', '00:00')
                schedule.every().day.at(time_str).do(
                    self._execute_job, name, func
                ).tag(name)
            
            elif schedule_type == "weekly":
                day = job_config.get('day', 'monday')
                time_str = job_config.get('time', '00:00')
                getattr(schedule.every(), day).at(time_str).do(
                    self._execute_job, name, func
                ).tag(name)
            
            elif schedule_type == "hourly":
                schedule.every().hour.do(
                    self._execute_job, name, func
                ).tag(name)
            
            logger.debug(f"Scheduled job: {name}")
            
        except Exception as e:
            logger.error(f"Error scheduling job {job_config['name']}: {e}")
    
    def _execute_job(self, job_name: str, func: Callable):
        """Execute a scheduled job"""
        start_time = time.time()
        
        try:
            logger.info(f"Starting scheduled job: {job_name}")
            
            # Update job stats
            job_config = self._jobs.get(job_name, {})
            job_config['run_count'] = job_config.get('run_count', 0) + 1
            job_config['last_run'] = datetime.now(timezone.utc)
            
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                # Run async function in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(func())
                loop.close()
            else:
                result = func()
            
            # Update success stats
            job_config['success_count'] = job_config.get('success_count', 0) + 1
            self._stats['successful_jobs'] += 1
            
            execution_time = time.time() - start_time
            logger.info(f"Completed job {job_name} in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            # Update failure stats
            job_config = self._jobs.get(job_name, {})
            job_config['failure_count'] = job_config.get('failure_count', 0) + 1
            self._stats['failed_jobs'] += 1
            
            execution_time = time.time() - start_time
            logger.error(f"Job {job_name} failed after {execution_time:.2f}s: {e}")
        
        finally:
            self._stats['total_jobs_run'] += 1
            self._stats['last_run'] = datetime.now(timezone.utc)
    
    def start(self):
        """Start the batch scheduler"""
        if self._running:
            logger.warning("Scheduler is already running")
            return
        
        self._running = True
        self._stop_event.clear()
        
        # Start scheduler thread
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        
        logger.info("Batch Scheduler started")
    
    def stop(self):
        """Stop the batch scheduler"""
        if not self._running:
            logger.warning("Scheduler is not running")
            return
        
        self._running = False
        self._stop_event.set()
        
        # Wait for scheduler thread to finish
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=30)
        
        # Clear all scheduled jobs
        schedule.clear()
        
        logger.info("Batch Scheduler stopped")
    
    def _run_scheduler(self):
        """Main scheduler loop"""
        logger.info("Scheduler thread started")
        
        while self._running and not self._stop_event.is_set():
            try:
                # Run pending jobs
                schedule.run_pending()
                
                # Update next run time
                jobs = schedule.get_jobs()
                if jobs:
                    next_job = min(jobs, key=lambda x: x.next_run)
                    self._stats['next_run'] = next_job.next_run
                
                # Sleep for a short interval
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(60)  # Wait longer on error
        
        logger.info("Scheduler thread stopped")
    
    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self._running
    
    def get_job_status(self, job_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job"""
        return self._jobs.get(job_name)
    
    def get_all_jobs_status(self) -> Dict[str, Any]:
        """Get status of all jobs"""
        return dict(self._jobs)
    
    def enable_job(self, job_name: str) -> bool:
        """Enable a job"""
        try:
            if job_name in self._jobs:
                self._jobs[job_name]['enabled'] = True
                # Re-schedule the job
                schedule.clear(job_name)
                self._schedule_job(self._jobs[job_name])
                logger.info(f"Enabled job: {job_name}")
                return True
            else:
                logger.warning(f"Job not found: {job_name}")
                return False
        except Exception as e:
            logger.error(f"Error enabling job {job_name}: {e}")
            return False
    
    def disable_job(self, job_name: str) -> bool:
        """Disable a job"""
        try:
            if job_name in self._jobs:
                self._jobs[job_name]['enabled'] = False
                # Remove from schedule
                schedule.clear(job_name)
                logger.info(f"Disabled job: {job_name}")
                return True
            else:
                logger.warning(f"Job not found: {job_name}")
                return False
        except Exception as e:
            logger.error(f"Error disabling job {job_name}: {e}")
            return False
    
    def run_job_now(self, job_name: str) -> bool:
        """Run a job immediately"""
        try:
            if job_name in self._jobs:
                func = self._jobs[job_name]['func']
                self._execute_job(job_name, func)
                logger.info(f"Manually executed job: {job_name}")
                return True
            else:
                logger.warning(f"Job not found: {job_name}")
                return False
        except Exception as e:
            logger.error(f"Error running job {job_name}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics"""
        uptime = datetime.now(timezone.utc) - self._stats['uptime_start']
        
        return {
            **self._stats,
            'running': self._running,
            'uptime_seconds': uptime.total_seconds(),
            'total_jobs': len(self._jobs),
            'enabled_jobs': len([j for j in self._jobs.values() if j.get('enabled', False)]),
            'pending_jobs': len(schedule.get_jobs())
        }
    
    # Default scheduled job functions
    async def _run_pre_market_scan(self):
        """Pre-market catalyst scan to catch overnight news"""
        try:
            logger.info("Starting pre-market catalyst scan")
            
            # Focus on major indices and popular stocks for pre-market
            pre_market_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
                'SPY', 'QQQ', 'IWM', 'DIA'  # Major ETFs
            ]
            
            # Use RSS and financial APIs for fastest overnight news
            sources = ['rss', 'financial']
            
            # Process with higher limits for overnight scan
            results = await self.processor.process_ticker_batch(
                tickers=pre_market_tickers,
                sources=sources,
                limit_per_source=30
            )
            
            total_catalysts = sum(len(catalysts) for catalysts in results.values())
            logger.info(f"Pre-market scan completed: {total_catalysts} catalysts detected")
            
            return results
            
        except Exception as e:
            logger.error(f"Pre-market scan failed: {e}")
            return {}

    async def _run_high_frequency_fetch(self):
        """High-frequency data fetching for active monitoring"""
        try:
            logger.info("Starting high-frequency data fetch")
            
            # Get top active tickers for focused monitoring
            # This would typically come from a watchlist or trending analysis
            active_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            
            # Process each ticker
            results = await self.processor.process_ticker_batch(
                tickers=active_tickers,
                sources=['newsapi', 'twitter', 'reddit'],
                limit_per_source=10
            )
            
            logger.info(f"High-frequency fetch completed: {sum(len(r) for r in results.values())} items processed")
            
        except Exception as e:
            logger.error(f"High-frequency fetch failed: {e}")
            raise
    
    async def _run_daily_comprehensive_scan(self):
        """Daily comprehensive scan of all configured sources"""
        try:
            logger.info("Starting daily comprehensive scan")
            
            # Get broader list of tickers for comprehensive analysis
            # This could be from S&P 500, NASDAQ 100, or user-defined watchlists
            comprehensive_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
                'NFLX', 'CRM', 'UBER', 'SPOT', 'SQ', 'PYPL', 'ZM', 'SHOP'
            ]
            
            # Process with all available sources
            results = await self.processor.process_ticker_batch(
                tickers=comprehensive_tickers,
                sources=['newsapi', 'twitter', 'reddit', 'rss', 'regulatory', 'financial'],
                limit_per_source=25
            )
            
            # Generate daily report
            total_catalysts = sum(len(r) for r in results.values())
            logger.info(f"Daily comprehensive scan completed: {total_catalysts} catalysts detected")
            
            # Optionally send daily summary alert
            if total_catalysts > 0:
                await self._send_daily_summary(results, total_catalysts)
            
        except Exception as e:
            logger.error(f"Daily comprehensive scan failed: {e}")
            raise
    
    async def _run_trending_analysis(self):
        """Analyze trending topics and catalysts"""
        try:
            logger.info("Starting trending analysis")
            
            # This would analyze recent data to identify trending topics
            trending_results = await self.processor.analyze_trending_catalysts(
                time_window_hours=24,
                min_mentions=3
            )
            
            logger.info(f"Trending analysis completed: {len(trending_results)} trending topics")
            
        except Exception as e:
            logger.error(f"Trending analysis failed: {e}")
            raise
    
    async def _run_weekly_cleanup(self):
        """Weekly maintenance and cleanup"""
        try:
            logger.info("Starting weekly cleanup")
            
            # Clean up old data
            cleanup_results = await self.processor.cleanup_old_data(days_old=30)
            
            # Clear old caches
            cache_cleared = await self.processor.clear_expired_caches()
            
            # Generate maintenance report
            logger.info(f"Weekly cleanup completed: {cleanup_results} records cleaned, {cache_cleared} caches cleared")
            
        except Exception as e:
            logger.error(f"Weekly cleanup failed: {e}")
            raise
    
    async def _send_daily_summary(self, results: Dict[str, List], total_catalysts: int):
        """Send daily summary alert"""
        try:
            # This would format and send a daily summary
            # For now, just log the summary
            logger.info(f"Daily Summary: {total_catalysts} catalysts detected across {len(results)} tickers")
            
            # In a full implementation, this would:
            # 1. Format a comprehensive daily report
            # 2. Send via configured alert channels
            # 3. Include top catalysts, trending topics, etc.
            
        except Exception as e:
            logger.error(f"Failed to send daily summary: {e}")
    
    async def _run_breaking_news_scan(self):
        """Real-time breaking news monitoring for immediate market-moving events"""
        try:
            logger.info("Starting breaking news scan")
            
            # Focus on high-priority news sources for breaking news
            priority_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META']
            
            # Use only fastest news sources for breaking news
            results = await self.processor.process_ticker_batch(
                tickers=priority_tickers,
                sources=['newsapi', 'rss'],  # Fastest, most reliable sources
                limit_per_source=5,
                priority_mode=True
            )
            
            # Process results for immediate alerts
            total_catalysts = sum(len(r) for r in results.values())
            if total_catalysts > 0:
                logger.info(f"Breaking news scan: {total_catalysts} new catalysts detected")
            
        except Exception as e:
            logger.error(f"Breaking news scan failed: {e}")
            raise
    
    async def _run_market_hours_intensive_scan(self):
        """Intensive monitoring during market hours for professional trading"""
        try:
            # Only run during market hours (9:30 AM - 4:00 PM ET)
            current_time = datetime.now(timezone.utc)
            market_open = current_time.replace(hour=14, minute=30, second=0)  # 9:30 AM ET in UTC
            market_close = current_time.replace(hour=21, minute=0, second=0)   # 4:00 PM ET in UTC
            
            if not (market_open <= current_time <= market_close):
                logger.debug("Market hours intensive scan skipped - market closed")
                return
            
            logger.info("Starting market hours intensive scan")
            
            # Focus on most active stocks during market hours
            active_tickers = [
                'SPY', 'QQQ', 'IWM',  # Major ETFs
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META',  # Mega caps
                'AMD', 'NFLX', 'CRM', 'UBER', 'SQ', 'PYPL'  # High beta stocks
            ]
            
            # Use all available sources for comprehensive coverage
            results = await self.processor.process_ticker_batch(
                tickers=active_tickers,
                sources=['newsapi', 'twitter', 'reddit', 'rss', 'financial'],
                limit_per_source=10,
                priority_mode=True
            )
            
            total_catalysts = sum(len(r) for r in results.values())
            logger.info(f"Market hours intensive scan: {total_catalysts} catalysts detected")
            
        except Exception as e:
            logger.error(f"Market hours intensive scan failed: {e}")
            raise
    
    async def _run_earnings_realtime_scan(self):
        """Real-time earnings monitoring during earnings season"""
        try:
            # This would be more sophisticated in production, checking earnings calendar
            current_time = datetime.now(timezone.utc)
            
            # More active during earnings season (typically quarterly: Jan, Apr, Jul, Oct)
            earnings_months = [1, 4, 7, 10]
            is_earnings_season = current_time.month in earnings_months
            
            if not is_earnings_season:
                logger.debug("Earnings real-time scan skipped - not earnings season")
                return
            
            logger.info("Starting earnings real-time scan")
            
            # Focus on companies likely to report earnings
            earnings_focus_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                'NFLX', 'AMD', 'CRM', 'UBER', 'SQ', 'PYPL', 'ZM'
            ]
            
            # Use earnings-focused sources
            results = await self.processor.process_ticker_batch(
                tickers=earnings_focus_tickers,
                sources=['newsapi', 'financial', 'rss'],
                limit_per_source=8,
                focus_categories=['earnings'],
                priority_mode=True
            )
            
            total_catalysts = sum(len(r) for r in results.values())
            if total_catalysts > 0:
                logger.info(f"Earnings real-time scan: {total_catalysts} earnings-related catalysts detected")
            
        except Exception as e:
            logger.error(f"Earnings real-time scan failed: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup scheduler resources"""
        try:
            self.stop()
            await self.processor.cleanup()
            logger.info("Batch Scheduler cleanup completed")
        except Exception as e:
            logger.error(f"Error during scheduler cleanup: {e}")
