"""
J.A.R.V.I.S. Event Manager
Advanced event handling and notification system
"""

import time
import threading
import queue
import uuid
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from enum import Enum


class EventPriority(Enum):
    """Event priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventType(Enum):
    """Event types"""
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    MODULE_LOADED = "module_loaded"
    MODULE_ERROR = "module_error"
    VOICE_COMMAND = "voice_command"
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_EVENT = "security_event"
    USER_ACTION = "user_action"
    NETWORK_EVENT = "network_event"
    FILE_EVENT = "file_event"
    CUSTOM = "custom"


class Event:
    """Event class for system events"""

    def __init__(self,
                 event_type: EventType,
                 data: Any = None,
                 priority: EventPriority = EventPriority.NORMAL,
                 source: str = "unknown",
                 event_id: str = None):
        """
        Initialize event

        Args:
            event_type: Type of event
            data: Event data payload
            priority: Event priority
            source: Source of the event
            event_id: Unique event identifier
        """
        self.event_id = event_id or str(uuid.uuid4())
        self.event_type = event_type
        self.data = data
        self.priority = priority
        self.source = source
        self.timestamp = time.time()
        self.created_at = datetime.now().isoformat()
        self.processed = False
        self.processing_time = None

    def __str__(self) -> str:
        return f"Event({self.event_type.value}, {self.priority.name}, {self.source})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "data": self.data,
            "priority": self.priority.name,
            "source": self.source,
            "timestamp": self.timestamp,
            "created_at": self.created_at,
            "processed": self.processed,
            "processing_time": self.processing_time
        }


class EventManager:
    """
    Advanced event management system
    Handles event routing, queuing, and processing
    """

    def __init__(self, jarvis_instance):
        """
        Initialize event manager

        Args:
            jarvis_instance: Reference to main JARVIS instance
        """
        self.jarvis = jarvis_instance

        # Event queues by priority
        self.event_queues = {
            EventPriority.LOW: queue.Queue(),
            EventPriority.NORMAL: queue.Queue(),
            EventPriority.HIGH: queue.Queue(),
            EventPriority.CRITICAL: queue.Queue()
        }

        # Event handlers
        self.event_handlers = {}  # event_type -> [handlers]
        self.global_handlers = []  # Handlers for all events

        # Event statistics
        self.event_stats = {
            "total_events": 0,
            "processed_events": 0,
            "failed_events": 0,
            "average_processing_time": 0.0
        }

        # Processing control
        self.is_processing = False
        self.processor_thread = None
        self.stop_processing = threading.Event()

        # Event history
        self.event_history = []
        self.max_history_size = 10000

    def start_processing(self):
        """Start event processing"""
        if not self.is_processing:
            self.is_processing = True
            self.stop_processing.clear()
            self.processor_thread = threading.Thread(
                target=self._process_events_loop,
                name="EventProcessor",
                daemon=True
            )
            self.processor_thread.start()
            self.jarvis.logger.info("Event processing started")

    def stop_processing(self):
        """Stop event processing"""
        if self.is_processing:
            self.is_processing = False
            self.stop_processing.set()

            if self.processor_thread and self.processor_thread.is_alive():
                self.processor_thread.join(timeout=5)

            self.jarvis.logger.info("Event processing stopped")

    def emit_event(self,
                   event_type: EventType,
                   data: Any = None,
                   priority: EventPriority = EventPriority.NORMAL,
                   source: str = "system") -> str:
        """
        Emit an event

        Args:
            event_type: Type of event to emit
            data: Event data
            priority: Event priority
            source: Event source

        Returns:
            Event ID
        """
        event = Event(event_type, data, priority, source)

        # Add to appropriate queue
        self.event_queues[priority].put(event)

        # Update statistics
        self.event_stats["total_events"] += 1

        # Add to history
        self._add_to_history(event)

        self.jarvis.logger.debug(f"Event emitted: {event}")

        return event.event_id

    def register_handler(self,
                        event_type: EventType,
                        handler: Callable,
                        priority: EventPriority = EventPriority.NORMAL):
        """
        Register event handler

        Args:
            event_type: Event type to handle
            handler: Handler function
            priority: Handler priority (for ordering)
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []

        handler_info = {
            "handler": handler,
            "priority": priority,
            "registered_at": time.time()
        }

        self.event_handlers[event_type].append(handler_info)

        # Sort handlers by priority
        self.event_handlers[event_type].sort(
            key=lambda x: x["priority"].value,
            reverse=True
        )

        self.jarvis.logger.debug(f"Handler registered for {event_type.value}")

    def register_global_handler(self, handler: Callable):
        """
        Register global event handler (handles all events)

        Args:
            handler: Global handler function
        """
        self.global_handlers.append({
            "handler": handler,
            "registered_at": time.time()
        })

    def unregister_handler(self, event_type: EventType, handler: Callable):
        """
        Unregister event handler

        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if event_type in self.event_handlers:
            self.event_handlers[event_type] = [
                h for h in self.event_handlers[event_type]
                if h["handler"] != handler
            ]

            if not self.event_handlers[event_type]:
                del self.event_handlers[event_type]

    def get_stats(self) -> Dict[str, Any]:
        """Get event processing statistics"""
        return {
            **self.event_stats,
            "queue_sizes": {
                priority.name: self.event_queues[priority].qsize()
                for priority in EventPriority
            },
            "registered_handlers": len(self.event_handlers),
            "global_handlers": len(self.global_handlers),
            "history_size": len(self.event_history),
            "is_processing": self.is_processing
        }

    def _process_events_loop(self):
        """Main event processing loop"""
        self.jarvis.logger.info("Event processing loop started")

        while not self.stop_processing.is_set():
            try:
                # Process events by priority (critical first)
                for priority in [EventPriority.CRITICAL, EventPriority.HIGH,
                               EventPriority.NORMAL, EventPriority.LOW]:

                    if self.stop_processing.is_set():
                        break

                    # Process up to 10 events per priority level
                    for _ in range(min(10, self.event_queues[priority].qsize())):
                        if self.stop_processing.is_set():
                            break

                        try:
                            event = self.event_queues[priority].get_nowait()
                            self._process_event(event)
                            self.event_queues[priority].task_done()

                        except queue.Empty:
                            break
                        except Exception as e:
                            self.jarvis.logger.error(f"Error processing event: {e}")
                            self.event_stats["failed_events"] += 1

                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)

            except Exception as e:
                self.jarvis.logger.error(f"Error in event processing loop: {e}")
                time.sleep(1)

        self.jarvis.logger.info("Event processing loop ended")

    def _process_event(self, event: Event):
        """
        Process a single event

        Args:
            event: Event to process
        """
        start_time = time.time()

        try:
            # Call global handlers first
            for handler_info in self.global_handlers:
                try:
                    handler_info["handler"](event)
                except Exception as e:
                    self.jarvis.logger.error(f"Error in global handler: {e}")

            # Call specific event handlers
            if event.event_type in self.event_handlers:
                for handler_info in self.event_handlers[event.event_type]:
                    try:
                        handler_info["handler"](event)
                    except Exception as e:
                        self.jarvis.logger.error(f"Error in event handler: {e}")

            # Mark as processed
            event.processed = True
            event.processing_time = time.time() - start_time

            # Update statistics
            self.event_stats["processed_events"] += 1
            total_time = self.event_stats["average_processing_time"] * (self.event_stats["processed_events"] - 1)
            self.event_stats["average_processing_time"] = (total_time + event.processing_time) / self.event_stats["processed_events"]

        except Exception as e:
            self.jarvis.logger.error(f"Error processing event {event.event_id}: {e}")
            self.event_stats["failed_events"] += 1

    def _add_to_history(self, event: Event):
        """Add event to history"""
        self.event_history.append(event.to_dict())

        # Maintain history size
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)

    def clear_history(self):
        """Clear event history"""
        self.event_history.clear()
        self.jarvis.logger.info("Event history cleared")

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent events from history

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent events
        """
        return self.event_history[-limit:] if self.event_history else []

    def get_events_by_type(self, event_type: EventType, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get events by type

        Args:
            event_type: Event type to filter by
            limit: Maximum number of events to return

        Returns:
            List of events of specified type
        """
        filtered_events = [
            event for event in self.event_history
            if event["event_type"] == event_type.value
        ]
        return filtered_events[-limit:]

    def process_events(self):
        """Process events (called from main loop)"""
        # This method can be used for synchronous event processing
        # or for triggering immediate processing of critical events
        pass

    def create_custom_event(self,
                           event_name: str,
                           data: Any = None,
                           priority: EventPriority = EventPriority.NORMAL) -> str:
        """
        Create and emit a custom event

        Args:
            event_name: Custom event name
            data: Event data
            priority: Event priority

        Returns:
            Event ID
        """
        # Create a temporary event type for custom events
        custom_event_type = EventType.CUSTOM

        # Store custom event info in data
        custom_data = {
            "custom_event_name": event_name,
            "original_data": data
        }

        return self.emit_event(custom_event_type, custom_data, priority, "custom")

    def schedule_event(self,
                      delay: float,
                      event_type: EventType,
                      data: Any = None,
                      priority: EventPriority = EventPriority.NORMAL) -> str:
        """
        Schedule an event to be emitted after a delay

        Args:
            delay: Delay in seconds
            event_type: Event type
            data: Event data
            priority: Event priority

        Returns:
            Event ID
        """
        def delayed_emit():
            time.sleep(delay)
            if not self.stop_processing.is_set():
                self.emit_event(event_type, data, priority, "scheduler")

        thread = threading.Thread(
            target=delayed_emit,
            name=f"ScheduledEvent-{event_type.value}",
            daemon=True
        )
        thread.start()

        return f"scheduled-{event_type.value}-{time.time()}"