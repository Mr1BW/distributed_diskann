/** Project: https://github.com/RamseyK/ByteBufferCpp */
/**
 ByteBuffer
 ByteBuffer.cpp
 Copyright 2011 - 2013 Ramsey Kant
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 Modfied 2015 by Ashley Davis (SgtCoDFish)
 */

 #include "byte_buffer.hpp"

 #ifdef BB_USE_NS
 namespace bb {
 #endif
 
 /**
  * ByteBuffer constructor
  * Reserves specified size in internal vector
  *
  * @param size Size (in bytes) of space to preallocate internally. Default is set in DEFAULT_SIZE
  */
 ByteBuffer::ByteBuffer(uint64_t size) {
     buf.reserve(size);
     clear();
 #ifdef BB_UTILITY
     name = "";
 #endif
 }
 
 /**
  * ByteBuffer constructor
  * Consume an entire uint8_t array of length len in the ByteBuffer
  *
  * @param arr uint8_t array of data (should be of length len)
  * @param size Size of space to allocate
  */
 ByteBuffer::ByteBuffer(const uint8_t* arr, uint64_t size) {
     // If the provided array is NULL, allocate a blank buffer of the provided size
     if (arr == NULL) {
         buf.reserve(size);
         clear();
     } else { // Consume the provided array
         buf.reserve(size);
         clear();
         putBytes(arr, size);
     }
 
 #ifdef BB_UTILITY
     name = "";
 #endif
 }
 
 /**
  * Bytes Remaining
  * Returns the number of bytes from the current read position till the end of the buffer
  *
  * @return Number of bytes from rpos to the end (size())
  */
 uint64_t ByteBuffer::bytesRemaining() {
     return size() - rpos;
 }
 
 /**
  * Clear
  * Clears out all data from the internal vector (original preallocated size remains), resets the positions to 0
  */
 void ByteBuffer::clear() {
     rpos = 0;
     wpos = 0;
     buf.clear();
 }
 
 /**
  * Clone
  * Allocate an exact copy of the ByteBuffer on the heap and return a pointer
  *
  * @return A pointer to the newly cloned ByteBuffer. NULL if no more memory available
  */
 std::unique_ptr<ByteBuffer> ByteBuffer::clone() {
     std::unique_ptr<ByteBuffer> ret = std::make_unique<ByteBuffer>(buf.size());
 
     // Copy data
     for (uint64_t i = 0; i < buf.size(); i++) {
         ret->put((uint8_t) get(i));
     }
 
     // Reset positions
     ret->setReadPos(0);
     ret->setWritePos(0);
 
     return ret;
 }
 
 /**
  * Equals, test for data equivilancy
  * Compare this ByteBuffer to another by looking at each byte in the internal buffers and making sure they are the same
  *
  * @param other A pointer to a ByteBuffer to compare to this one
  * @return True if the internal buffers match. False if otherwise
  */
 bool ByteBuffer::equals(ByteBuffer* other) {
     // If sizes aren't equal, they can't be equal
     if (size() != other->size())
         return false;
 
     // Compare byte by byte
     uint64_t len = size();
     for (uint64_t i = 0; i < len; i++) {
         if ((uint8_t) get(i) != (uint8_t) other->get(i))
             return false;
     }
 
     return true;
 }
 
 /**
  * Resize
  * Reallocates memory for the internal buffer of size newSize. Read and write positions will also be reset
  *
  * @param newSize The amount of memory to allocate
  */
 void ByteBuffer::resize(uint64_t newSize) {
     buf.resize(newSize);
     rpos = 0;
     wpos = 0;
 }
 
 /**
  * Size
  * Returns the size of the internal buffer...not necessarily the length of bytes used as data!
  *
  * @return size of the internal buffer
  */
 uint64_t ByteBuffer::size() const{
     return buf.size();
 }
 
 // Replacement
 
 /**
  * Replace
  * Replace occurance of a particular uint8_t, key, with the uint8_t rep
  *
  * @param key uint8_t to find for replacement
  * @param rep uint8_t to replace the found key with
  * @param start Index to start from. By default, start is 0
  * @param firstOccuranceOnly If true, only replace the first occurance of the key. If false, replace all occurances. False by default
  */
 void ByteBuffer::replace(uint8_t key, uint8_t rep, uint64_t start, bool firstOccuranceOnly) {
     uint64_t len = buf.size();
     for (uint64_t i = start; i < len; i++) {
         uint8_t data = read<uint8_t>(i);
         // Wasn't actually found, bounds of buffer were exceeded
         if ((key != 0) && (data == 0))
             break;
 
         // Key was found in array, perform replacement
         if (data == key) {
             buf[i] = rep;
             if (firstOccuranceOnly)
                 return;
         }
     }
 }
 
 // Read Functions
 
 uint8_t ByteBuffer::peek() const {
     return read<uint8_t>(rpos);
 }
 
 uint8_t ByteBuffer::get() const {
     return read<uint8_t>();
 }
 
 uint8_t ByteBuffer::get(uint64_t index) const {
     return read<uint8_t>(index);
 }
 
 void ByteBuffer::getBytes(uint8_t* out_buf, uint64_t len) const {
    if (rpos + len <= buf.size()) {
        memcpy(out_buf, &buf[rpos], len);
        rpos += len;
    }
 }

 void ByteBuffer::getBytes(const uint64_t len, ByteBuffer &out_buf) const {
    if (rpos + len <= this->buf.size()) {
        out_buf.putBytes(&this->buf[rpos], len);
        rpos += len;
    }
 }
 
 char ByteBuffer::getChar() const {
     return read<char>();
 }
 
 char ByteBuffer::getChar(uint64_t index) const {
     return read<char>(index);
 }
 
 double ByteBuffer::getDouble() const {
     return read<double>();
 }
 
 double ByteBuffer::getDouble(uint64_t index) const {
     return read<double>(index);
 }
 
 float ByteBuffer::getFloat() const {
     return read<float>();
 }
 
 float ByteBuffer::getFloat(uint64_t index) const {
     return read<float>(index);
 }
 
 int32_t ByteBuffer::getInt() const {
     return read<int32_t>();
 }
 
 int32_t ByteBuffer::getInt(uint64_t index) const {
     return read<int32_t>(index);
 }
 
 int64_t ByteBuffer::getLong() const {
     return read<int64_t>();
 }
 
 int64_t ByteBuffer::getLong(uint64_t index) const {
     return read<int64_t>(index);
 }
 
 uint16_t ByteBuffer::getShort() const {
     return read<uint16_t>();
 }
 
 uint16_t ByteBuffer::getShort(uint64_t index) const {
     return read<uint16_t>(index);
 }
 
 // Write Functions
 
 void ByteBuffer::put(ByteBuffer* src) {
     uint64_t len = src->size();
     for (uint64_t i = 0; i < len; i++)
         append<uint8_t>(src->get(i));
 }
 
 void ByteBuffer::put(uint8_t b) {
     append<uint8_t>(b);
 }
 
 void ByteBuffer::put(uint8_t b, uint64_t index) {
     insert<uint8_t>(b, index);
 }
 
 void ByteBuffer::putBytes(const uint8_t* b, const uint64_t len) {
        uint64_t s = len;
        if (size() < (wpos + s))
            buf.resize(wpos + s);
        memcpy(&buf[wpos], b, s);
        wpos += s;
 }
 
 void ByteBuffer::putBytes(uint8_t* b, uint64_t len, uint64_t index) {
     wpos = index;
 
     // Insert the data one byte at a time into the internal buffer at position i+starting index
     for (uint64_t i = 0; i < len; i++)
         append<uint8_t>(b[i]);
 }
 
 void ByteBuffer::putChar(char value) {
     append<char>(value);
 }
 
 void ByteBuffer::putChar(char value, uint64_t index) {
     insert<char>(value, index);
 }
 
 void ByteBuffer::putDouble(double value) {
     append<double>(value);
 }
 
 void ByteBuffer::putDouble(double value, uint64_t index) {
     insert<double>(value, index);
 }
 void ByteBuffer::putFloat(float value) {
     append<float>(value);
 }
 
 void ByteBuffer::putFloat(float value, uint64_t index) {
     insert<float>(value, index);
 }
 
 void ByteBuffer::putInt(int32_t value) {
     append<int32_t>(value);
 }
 
 void ByteBuffer::putInt(int32_t value, uint64_t index) {
     insert<int32_t>(value, index);
 }
 
 void ByteBuffer::putLong(int64_t value) {
     append<int64_t>(value);
 }
 
 void ByteBuffer::putLong(int64_t value, uint64_t index) {
     insert<int64_t>(value, index);
 }
 
 void ByteBuffer::putShort(uint16_t value) {
     append<uint16_t>(value);
 }
 
 void ByteBuffer::putShort(uint16_t value, uint64_t index) {
     insert<uint16_t>(value, index);
 }
 
 // Utility Functions
 #ifdef BB_UTILITY
 void ByteBuffer::setName(std::string n) {
     name = n;
 }
 
 std::string ByteBuffer::getName() {
     return name;
 }
 
 void ByteBuffer::printInfo() {
     uint64_t length = buf.size();
     std::cout << "ByteBuffer " << name.c_str() << " Length: " << length << ". Info Print" << std::endl;
 }
 
 void ByteBuffer::printAH() {
     uint64_t length = buf.size();
     std::cout << "ByteBuffer " << name.c_str() << " Length: " << length << ". ASCII & Hex Print" << std::endl;
 
     for (uint64_t i = 0; i < length; i++) {
         std::printf("0x%02x ", buf[i]);
     }
 
     std::printf("\n");
     for (uint64_t i = 0; i < length; i++) {
         std::printf("%c ", buf[i]);
     }
 
     std::printf("\n");
 }
 
 void ByteBuffer::printAscii() {
     uint64_t length = buf.size();
     std::cout << "ByteBuffer " << name.c_str() << " Length: " << length << ". ASCII Print" << std::endl;
 
     for (uint64_t i = 0; i < length; i++) {
         std::printf("%c ", buf[i]);
     }
 
     std::printf("\n");
 }
 
 void ByteBuffer::printHex() {
     uint64_t length = buf.size();
     std::cout << "ByteBuffer " << name.c_str() << " Length: " << length << ". Hex Print" << std::endl;
 
     for (uint64_t i = 0; i < length; i++) {
         std::printf("0x%02x ", buf[i]);
     }
 
     std::printf("\n");
 }
 
 void ByteBuffer::printPosition() {
     uint64_t length = buf.size();
     std::cout << "ByteBuffer " << name.c_str() << " Length: " << length << " Read Pos: " << rpos << ". Write Pos: "
             << wpos << std::endl;
 }
 
 #ifdef BB_USE_NS
 }
 #endif
 
 #endif