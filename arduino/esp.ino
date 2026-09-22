/*
 * ESP32-S3-WROOM-CAM — BLE photo + mode sender
 * ---------------------------------------------------------------
 * Single job for this firmware:
 *   - Button on GPIO4 (CAPTURE_BTN): capture a photo, resize it to
 *     224x224, JPEG-encode it, and send it over BLE to the connected
 *     phone.
 *   - Button on GPIO5 (MODE_BTN): cycle mode 1 -> 2 -> 3 -> 1 -> ...
 *     and send the current mode number over BLE.
 *
 * !! WIRING WARNING !!
 * On the ESP32-S3-WROOM-CAM pinout below (same one from your original
 * sketch), GPIO4 = SIOD (camera SDA) and GPIO5 = SIOC (camera SCL) —
 * these are the camera's I2C control lines. Wiring buttons to the same
 * pins will conflict with the camera and it will likely fail to
 * initialize (or behave erratically). If your board really has the
 * camera hardwired to GPIO4/5, move the buttons to two other free
 * GPIOs (e.g. GPIO2 / GPIO3) instead — the code below is written so
 * you only need to change CAPTURE_BTN_PIN / MODE_BTN_PIN.
 * ---------------------------------------------------------------
 */

#include "esp_camera.h"
#include "img_converters.h"
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

// ================== ESP32-S3-WROOM-CAM camera pinout ==================
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM      4
#define SIOC_GPIO_NUM      5

#define Y9_GPIO_NUM       16
#define Y8_GPIO_NUM       17
#define Y7_GPIO_NUM       18
#define Y6_GPIO_NUM       12
#define Y5_GPIO_NUM       10
#define Y4_GPIO_NUM        8
#define Y3_GPIO_NUM        9
#define Y2_GPIO_NUM       11

#define VSYNC_GPIO_NUM     6
#define HREF_GPIO_NUM      7
#define PCLK_GPIO_NUM     13
// ========================================================================

// ---- Buttons ----
#define CAPTURE_BTN_PIN   4     // take photo (see wiring warning above)
#define MODE_BTN_PIN      5     // cycle mode (see wiring warning above)
#define DEBOUNCE_MS       200

// ---- Output image size ----
#define OUT_W  224
#define OUT_H  224

// ---- BLE ----
#define SERVICE_UUID          "12345678-0000-1000-8000-00805f9b34fb"
#define PHOTO_META_UUID       "12345678-0001-1000-8000-00805f9b34fb" // notify: 4-byte total length (uint32 LE)
#define PHOTO_DATA_UUID       "12345678-0002-1000-8000-00805f9b34fb" // notify: [2-byte seq LE][chunk bytes]
#define MODE_CHAR_UUID        "12345678-0003-1000-8000-00805f9b34fb" // notify: 1 byte mode (1/2/3)

#define BLE_CHUNK_SIZE     200   // bytes of image data per notification
                                  // (phone must negotiate MTU >= ~230 for this to be efficient)

BLECharacteristic *photoMetaChar;
BLECharacteristic *photoDataChar;
BLECharacteristic *modeChar;

bool deviceConnected = false;
uint8_t currentMode = 1;

static camera_config_t camera_config = {
    .pin_pwdn = PWDN_GPIO_NUM,
    .pin_reset = RESET_GPIO_NUM,
    .pin_xclk = XCLK_GPIO_NUM,
    .pin_sscb_sda = SIOD_GPIO_NUM,
    .pin_sscb_scl = SIOC_GPIO_NUM,

    .pin_d7 = Y9_GPIO_NUM,
    .pin_d6 = Y8_GPIO_NUM,
    .pin_d5 = Y7_GPIO_NUM,
    .pin_d4 = Y6_GPIO_NUM,
    .pin_d3 = Y5_GPIO_NUM,
    .pin_d2 = Y4_GPIO_NUM,
    .pin_d1 = Y3_GPIO_NUM,
    .pin_d0 = Y2_GPIO_NUM,
    .pin_vsync = VSYNC_GPIO_NUM,
    .pin_href = HREF_GPIO_NUM,
    .pin_pclk = PCLK_GPIO_NUM,

    .xclk_freq_hz = 20000000,
    .ledc_timer = LEDC_TIMER_0,
    .ledc_channel = LEDC_CHANNEL_0,

    .pixel_format = PIXFORMAT_JPEG,
    .frame_size = FRAMESIZE_QVGA,          // 320x240 captured, then resized to 224x224
    .jpeg_quality = 12,
    .fb_count = 1,
    .fb_location = CAMERA_FB_IN_PSRAM,
    .grab_mode = CAMERA_GRAB_WHEN_EMPTY,
};

class ServerCallbacks : public BLEServerCallbacks {
    void onConnect(BLEServer* pServer) override {
        deviceConnected = true;
        Serial.println("BLE client connected");
    }
    void onDisconnect(BLEServer* pServer) override {
        deviceConnected = false;
        Serial.println("BLE client disconnected");
        BLEDevice::startAdvertising();  // resume advertising so phone can reconnect
    }
};

bool cameraInit() {
    esp_err_t err = esp_camera_init(&camera_config);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed: 0x%x\n", err);
        return false;
    }
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_vflip(s, 1);
        s->set_hmirror(s, 1);
    }
    return true;
}

// Nearest-neighbor resize, RGB888, stretches src -> dstW x dstH (no cropping)
void resizeRGB888(const uint8_t *src, int srcW, int srcH, uint8_t *dst, int dstW, int dstH) {
    for (int y = 0; y < dstH; y++) {
        int sy = y * srcH / dstH;
        for (int x = 0; x < dstW; x++) {
            int sx = x * srcW / dstW;
            const uint8_t *sp = src + (sy * srcW + sx) * 3;
            uint8_t *dp = dst + (y * dstW + x) * 3;
            dp[0] = sp[0];
            dp[1] = sp[1];
            dp[2] = sp[2];
        }
    }
}

// Capture -> resize to 224x224 -> JPEG-encode -> send over BLE
void captureAndSendPhoto() {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Capture failed");
        return;
    }

    uint8_t *rgbBuf = (uint8_t*)malloc(fb->width * fb->height * 3);
    if (!rgbBuf) {
        Serial.println("rgbBuf alloc failed");
        esp_camera_fb_return(fb);
        return;
    }
    bool ok = fmt2rgb888(fb->buf, fb->len, PIXFORMAT_JPEG, rgbBuf);
    int srcW = fb->width, srcH = fb->height;
    esp_camera_fb_return(fb);

    if (!ok) {
        Serial.println("JPEG->RGB888 conversion failed");
        free(rgbBuf);
        return;
    }

    uint8_t *resizedBuf = (uint8_t*)malloc(OUT_W * OUT_H * 3);
    if (!resizedBuf) {
        Serial.println("resizedBuf alloc failed");
        free(rgbBuf);
        return;
    }
    resizeRGB888(rgbBuf, srcW, srcH, resizedBuf, OUT_W, OUT_H);
    free(rgbBuf);

    uint8_t *jpgBuf = nullptr;
    size_t jpgLen = 0;
    bool jpgOk = fmt2jpg(resizedBuf, OUT_W * OUT_H * 3, OUT_W, OUT_H, PIXFORMAT_RGB888, 80, &jpgBuf, &jpgLen);
    free(resizedBuf);

    if (!jpgOk) {
        Serial.println("RGB888->JPEG encode failed");
        return;
    }

    if (!deviceConnected) {
        Serial.println("No BLE client connected, discarding photo");
        free(jpgBuf);
        return;
    }

    // 1. Send total length so the phone knows when the transfer is done
    uint32_t totalLen = (uint32_t)jpgLen;
    photoMetaChar->setValue((uint8_t*)&totalLen, 4);
    photoMetaChar->notify();
    delay(20);

    // 2. Send chunks: [2-byte seq LE][data...]
    uint16_t seq = 0;
    size_t offset = 0;
    uint8_t packet[2 + BLE_CHUNK_SIZE];
    while (offset < jpgLen) {
        size_t chunkLen = min((size_t)BLE_CHUNK_SIZE, jpgLen - offset);
        packet[0] = seq & 0xFF;
        packet[1] = (seq >> 8) & 0xFF;
        memcpy(packet + 2, jpgBuf + offset, chunkLen);

        photoDataChar->setValue(packet, chunkLen + 2);
        photoDataChar->notify();

        offset += chunkLen;
        seq++;
        delay(15); // give the BLE stack time to flush its notify queue
    }

    Serial.printf("Sent photo: %u bytes in %u chunks\n", (unsigned)jpgLen, seq);
    free(jpgBuf);
}

void sendMode() {
    if (!deviceConnected) {
        Serial.println("No BLE client connected, mode not sent");
        return;
    }
    modeChar->setValue(&currentMode, 1);
    modeChar->notify();
    Serial.printf("Mode -> %d\n", currentMode);
}

void setup() {
    Serial.begin(115200);
    delay(500);

    pinMode(CAPTURE_BTN_PIN, INPUT_PULLUP);
    pinMode(MODE_BTN_PIN, INPUT_PULLUP);

    if (!cameraInit()) {
        Serial.println("Camera init failed, halting");
        while (true) delay(1000);
    }
    Serial.println("Camera OK");

    BLEDevice::init("BanknoteReader");
    BLEServer *pServer = BLEDevice::createServer();
    pServer->setCallbacks(new ServerCallbacks());

    BLEService *pService = pServer->createService(SERVICE_UUID);

    photoMetaChar = pService->createCharacteristic(
        PHOTO_META_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    photoMetaChar->addDescriptor(new BLE2902());

    photoDataChar = pService->createCharacteristic(
        PHOTO_DATA_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    photoDataChar->addDescriptor(new BLE2902());

    modeChar = pService->createCharacteristic(
        MODE_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
    modeChar->addDescriptor(new BLE2902());

    pService->start();

    BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();
    pAdvertising->addServiceUUID(SERVICE_UUID);
    pAdvertising->setScanResponse(true);
    BLEDevice::startAdvertising();

    Serial.println("BLE advertising started");
}

void loop() {
    static unsigned long lastCaptureMs = 0;
    static unsigned long lastModeMs = 0;

    if (digitalRead(CAPTURE_BTN_PIN) == LOW && millis() - lastCaptureMs > DEBOUNCE_MS) {
        lastCaptureMs = millis();
        captureAndSendPhoto();
    }

    if (digitalRead(MODE_BTN_PIN) == LOW && millis() - lastModeMs > DEBOUNCE_MS) {
        lastModeMs = millis();
        currentMode = (currentMode % 3) + 1;  // 1 -> 2 -> 3 -> 1 ...
        sendMode();
    }

    delay(10);
}
