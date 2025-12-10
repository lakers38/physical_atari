# Copyright 2025 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# apt-get install libusb-1.0-0-dev
# pip install libusb1

import threading

import usb1

from framework.Logger import logger


class HIDDevice:
    def __init__(self, vendor_id_str: str, product_id_str: str, endpoint_out=0x01):
        self.vendor_id = int(vendor_id_str, 16)
        self.product_id = int(product_id_str, 16)
        self.endpoint_out = endpoint_out

        self.context = usb1.USBContext()
        self.handle = self.context.openByVendorIDAndProductID(self.vendor_id, self.product_id)
        if self.handle is None:
            raise ValueError(f"Device {vendor_id_str}:{product_id_str} not found.")
        self.handle.claimInterface(0)

        self._running = True
        self._event_thread = threading.Thread(target=self._handle_events, daemon=True)
        self._event_thread.start()

    def shutdown(self):
        self._running = False
        self._event_thread.join(timeout=1.0)
        try:
            self.handle.releaseInterface(0)
        except usb1.USBError as e:
            logger.warning(f"shutdown: failed to release interface: {e}")
        self.context.close()

    def write_sync(self, data: bytes, timeout=1000):
        # blocking write to the HID device
        try:
            self.handle.interruptWrite(self.endpoint_out, data, timeout)
        except usb1.USBError as e:
            logger.warning(f"write_sync: USB write failed: {e}")

    def write_async(self, data: bytes):
        # submit non-blocking write to HID device using interrupt endpoint
        def _async_callback(transfer):
            status = transfer.getStatus()
            if status != usb1.TRANSFER_COMPLETED:
                logger.warning(f"write_async: usb transfer failed with status {status}")
            transfer.close()

        transfer = self.handle.getTransfer()
        transfer.setInterrupt(endpoint=self.endpoint_out, data=data, callback=_async_callback, timeout=1000)
        try:
            transfer.submit()
        except usb1.USBError as e:
            logger.warning(f"write_async: async transfer failed to submit: {e}")
            transfer.close()

    def _handle_events(self):
        # Handle async USB events in a background thread
        while self._running:
            try:
                self.context.handleEventsTimeout(tv=0.01)
            except usb1.USBErrorInterrupted:
                continue  # expected on shutdown
            except Exception as e:
                logger.warning(f"_handle_events: error: {e}")
