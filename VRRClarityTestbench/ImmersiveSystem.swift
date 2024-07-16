//
//  ImmersiveSystem.swift
//  VRRClarityTestbench
//
//  Created by Max Thomas on 4/24/24.
//

import RealityKit
import ARKit
import QuartzCore
import Metal
import MetalKit
import Spatial
import AVFoundation
import SceneKit

// Render to the full headset FOV
let fullFOVRender = false

// Render texture params
let renderWidth = Int(fullFOVRender ? 1888+293 : 1920) // left/right eye are spaced 256px x 80px apart, so we adjust for that.
let renderHeight = Int(fullFOVRender ? 1824+84 : 1080) // 1824 for full screen
let renderScale = fullFOVRender ? 2.5 : 1.0

//
// Test suite parameters
// -----------------------------------------------------------------
// Headlock the test image including pitch/roll. If false, only headlock yaw.
let headlockTestImage = false

// Just place the image in the world, no headlocking
let imageDoesntFollowHeadAtAll = false

// Display different mipmap levels (below 1x) with a yellow -> orange -> red gradient
let colorMipLevels = true

// Level 0/1x shows as solid green, not a test image
let onlyColorsNoTestImage = false

// Test the test texture without mipmaps on, if false
let enableDrawableMipmaps = true

// How to filter the image when it is drawn by RealityKit
let imageFilteringMethod = ImageFilteringMethod.bicubic

// Virtual screen size/depth
let virtualScreenDepth: Float = 30.0 * inchesToMeters // 30in away
let virtualScreenDiagonal: Float = 28.0 * inchesToMeters

let colorMipmapLevelStart: Int = 1 // set to 2 for 4k textures, or to view the texture fully.

let testImageFilename = "857a4-2020-kgontech-1920x1080-tuff-test-white-on-black"
//let testImageFilename = "One-Pixel-Checkerboard-2024-001-copy"

//
// -----------------------------------------------------------------
//

//
// Misc consts
//
// Calculate screen width/height from diagonal
let diagonalAspectRatio = sqrt(pow(Float(renderWidth), 2) + pow(Float(renderHeight), 2))
let heightRatio = Float(renderHeight) / diagonalAspectRatio
let widthRatio = Float(renderWidth) / diagonalAspectRatio
let virtualScreenWidth: Float = widthRatio * virtualScreenDiagonal // 62cm, or 24.4in
let virtualScreenHeight: Float = heightRatio * virtualScreenDiagonal // 34.9cm, or 13.7in

let maxBuffersInFlight = 3
let maxPlanesDrawn = 1024
let renderFormat = MTLPixelFormat.bgra8Unorm_srgb // rgba8Unorm, rgba8Unorm_srgb, bgra8Unorm, bgra8Unorm_srgb, rgba16Float
let renderZNear = 0.001
let renderZFar = 100.0
let inchesToMeters: Float = 25.4 / 1000.0

enum ImageFilteringMethod {
    case nearest
    case bilinear
    case bicubic
}

class VisionPro: NSObject, ObservableObject {
    let arSession = ARKitSession()
    let worldTracking = WorldTrackingProvider()
    let handTracking = HandTrackingProvider()
    let sceneReconstruction = SceneReconstructionProvider()
    let planeDetection = PlaneDetectionProvider()
    var nextFrameTime: TimeInterval = 0.0
    
    var planeAnchors: [UUID: PlaneAnchor] = [:]
    var planeLock = NSObject()
    var vsyncDelta: Double = (1.0 / 90.0)
    
    override init() {
        super.init()
        self.createDisplayLink()
        
        Task {
            await processPlaneUpdates()
        }
    }
    
    func transformMatrix() -> simd_float4x4 {
        guard let deviceAnchor = worldTracking.queryDeviceAnchor(atTimestamp: nextFrameTime)
        else {
            print ("Failed to get anchor?")
            return .init()
        }
        return deviceAnchor.originFromAnchorTransform
    }
    
    func runArkitSession() async {
       let authStatus = await arSession.requestAuthorization(for: [.handTracking, .worldSensing])
        
        var trackingList: [any DataProvider] = [worldTracking]
        if authStatus[.handTracking] == .allowed {
            trackingList.append(handTracking)
        }
        if authStatus[.worldSensing] == .allowed {
            trackingList.append(sceneReconstruction)
            trackingList.append(planeDetection)
        }
        
        do {
            try await arSession.run(trackingList)
        } catch {
            fatalError("Failed to initialize ARSession")
        }
    }
    
    func createDisplayLink() {
        let displaylink = CADisplayLink(target: self, selector: #selector(frame))
        displaylink.add(to: .current, forMode: RunLoop.Mode.default)
    }
    
    @objc func frame(displaylink: CADisplayLink) {
        let frameDuration = displaylink.targetTimestamp - displaylink.timestamp
        nextFrameTime = displaylink.targetTimestamp + (frameDuration * 4)
        vsyncDelta = frameDuration
        //print("vsync frame", frameDuration, displaylink.targetTimestamp - CACurrentMediaTime(), displaylink.timestamp - CACurrentMediaTime())
    }
    
    func processPlaneUpdates() async {
        for await update in planeDetection.anchorUpdates {
            //print(update.event, update.anchor.classification, update.anchor.id, update.anchor.description)
            if update.anchor.classification == .window {
                // Skip planes that are windows.
                continue
            }
            switch update.event {
            case .added, .updated:
                updatePlane(update.anchor)
            case .removed:
                removePlane(update.anchor)
            }
        }
    }
    
    func updatePlane(_ anchor: PlaneAnchor) {
        lockPlaneAnchors()
        planeAnchors[anchor.id] = anchor
        unlockPlaneAnchors()
    }

    func removePlane(_ anchor: PlaneAnchor) {
        lockPlaneAnchors()
        planeAnchors.removeValue(forKey: anchor.id)
        unlockPlaneAnchors()
    }
    
    func lockPlaneAnchors() {
        objc_sync_enter(planeLock)
    }
    
    func unlockPlaneAnchors() {
         objc_sync_exit(planeLock)
    }
}

class ImmersiveSystem : System {
    let visionPro = VisionPro()
    var lastUpdateTime = 0.0
    var drawableQueue: TextureResource.DrawableQueue? = nil
    private(set) var surfaceMaterial: ShaderGraphMaterial? = nil
    private var textureResource: TextureResource?
    
    public let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var textureCache: CVMetalTextureCache!
    
    var currentRenderWidth: Int = Int(Double(renderWidth) * renderScale)
    var currentRenderHeight: Int = Int(Double(renderHeight) * renderScale)
    var currentRenderScale: Double = renderScale
    
    var mipColorTextures = [MTLTexture]()
    var testImageTexture: MTLTexture
    var lastFrameFetch: Double = 0.0
    
    required init(scene: RealityKit.Scene) {
        //visionPro.createDisplayLink()
        self.device = MTLCreateSystemDefaultDevice()!
        self.commandQueue = self.device.makeCommandQueue()!
        
        let desc = TextureResource.DrawableQueue.Descriptor(pixelFormat: renderFormat, width: currentRenderWidth, height: currentRenderHeight, usage: [.renderTarget, .shaderRead], mipmapsMode: enableDrawableMipmaps ? .allocateAll : .none)
        self.drawableQueue = try? TextureResource.DrawableQueue(desc)
        self.drawableQueue!.allowsNextDrawableTimeout = true
        
        let textureLoader = MTKTextureLoader(device: device)
        testImageTexture = try! textureLoader.newTexture(URL: Bundle.main.url(forResource: testImageFilename, withExtension: "png")!, options: [.generateMipmaps: true])
        
        let data = Data([0x00, 0x00, 0x00, 0xFF])
        self.textureResource = try! TextureResource(
            dimensions: .dimensions(width: 1, height: 1),
            format: .raw(pixelFormat: .bgra8Unorm),
            contents: .init(
                mipmapLevels: [
                    .mip(data: data, bytesPerRow: 4),
                ]
            )
        )

        Task {
            await visionPro.runArkitSession()
        }
        Task {
            var materialName = switch imageFilteringMethod {
                case .nearest:
                    "/Root/MonoMaterialNearest"
                case .bilinear:
                    "/Root/MonoMaterialBilinear"
                case .bicubic:
                    "/Root/MonoMaterialBicubic"
            }
            self.surfaceMaterial = try! await ShaderGraphMaterial(
                named: materialName,
                from: "SBSMaterial.usda"
            )
            try! self.surfaceMaterial!.setParameter(
                name: "texture",
                value: .textureResource(self.textureResource!)
            )
            textureResource!.replace(withDrawables: drawableQueue!)
        }
        
        if CVMetalTextureCacheCreate(nil, nil, self.device, nil, &textureCache) != 0 {
            fatalError("CVMetalTextureCacheCreate")
        }
        
        let colors = [
            MTLClearColor(red: 0.0, green: 1.0, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 1.0, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.75, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.5, blue: 0.1, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.25, blue: 0.1, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.125, blue: 0.1, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0), // 7
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0), // 11, this is where 1920x1080 goes to
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0),
            MTLClearColor(red: 1.0, green: 0.0, blue: 0.0, alpha: 1.0), // 15
        ]
        
        for i in 0..<16 {
            var size = CGSize(width: currentRenderWidth / (1<<i), height: (currentRenderHeight) / (1<<i))
            if size.width <= 0 {
                size.width = 1
            }
            if size.height <= 0 {
                size.height = 1
            }
            mipColorTextures.append(createTextureWithColor(color: colors[i], size: size)!)
        }
    }

    func update(context: SceneUpdateContext) {
        // RealityKit automatically calls this every frame for every scene.
        let plane = context.scene.findEntity(named: "video_plane") as? ModelEntity
        if let plane = plane {
            //print("frame", plane.id)
            
            let transform = visionPro.transformMatrix()
            
            var planeTransform = transform
            //planeTransform *= renderViewTransforms[0]
            if headlockTestImage || fullFOVRender {
                planeTransform.columns.3 -= transform.columns.2 * virtualScreenDepth
            }
            //planeTransform.columns.3 += transform.columns.0 * 0.5
            
            //planeTransform.columns.3 += DummyMetalRenderer.renderViewTransforms[0].columns.3
            
            do {
                if let surfaceMaterial = surfaceMaterial {
                    plane.model?.materials = [surfaceMaterial]
                }
                
                let drawable = try drawableQueue?.nextDrawable()
                if drawable == nil {
                    return
                }
                
                var scale = simd_float3(virtualScreenWidth, 1.0, virtualScreenHeight)
                if fullFOVRender {
                    scale = simd_float3(DummyMetalRenderer.renderTangents[0].x + DummyMetalRenderer.renderTangents[0].y, 1.0, DummyMetalRenderer.renderTangents[0].z + DummyMetalRenderer.renderTangents[0].w)
                    scale *= virtualScreenDepth
                }
                
                var orientation = simd_quatf(transform) * simd_quatf(angle: 1.5708, axis: simd_float3(1,0,0))
                var position = simd_float3(planeTransform.columns.3.x, planeTransform.columns.3.y, planeTransform.columns.3.z)
                if !headlockTestImage && !fullFOVRender {
                    orientation = RemovePitchAndRoll(orientation) * simd_quatf(angle: 1.5708, axis: simd_float3(1,0,0))
                    let forward = orientation.act(simd_float3(0.0, 1.0, 0.0))
                    position -= (forward * virtualScreenDepth)
                }
                
                //print(String(format: "%.2f, %.2f, %.2f", planeTransform.columns.3.x, planeTransform.columns.3.y, planeTransform.columns.3.z), CACurrentMediaTime() - lastUpdateTime)
                lastUpdateTime = CACurrentMediaTime()
                
                drawNextTexture(drawable: drawable!, simdDeviceAnchor: transform, plane: plane, position: position, orientation: orientation, scale: scale)
                drawable!.presentOnSceneUpdate()
            }
            catch {
            
            }
        }
    }
    
    func createTextureWithColor(color: MTLClearColor, size: CGSize) -> MTLTexture? {
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm_srgb,
                                                                         width: Int(size.width),
                                                                         height: Int(size.height),
                                                                         mipmapped: false)
        textureDescriptor.usage = [.shaderRead, .renderTarget]
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
            return nil
        }
        
        let region = MTLRegion(origin: MTLOrigin(x: 0, y: 0, z: 0),
                               size: MTLSize(width: Int(size.width), height: Int(size.height), depth: 1))
        
        var colorComponents: [UInt8] = [UInt8](repeating: 0, count: 4*Int(size.width)*Int(size.height))
        colorComponents[0] = UInt8(color.blue * 255)
        colorComponents[1] = UInt8(color.green * 255)
        colorComponents[2] = UInt8(color.red * 255)
        colorComponents[3] = UInt8(color.alpha * 255)
        
        for i in 0..<Int(size.width)*Int(size.height) {
            colorComponents[(i*4)+0] = colorComponents[0]
            colorComponents[(i*4)+1] = colorComponents[1]
            colorComponents[(i*4)+2] = colorComponents[2]
            colorComponents[(i*4)+3] = colorComponents[3]
        }
        
        texture.replace(region: region,
                        mipmapLevel: 0,
                        withBytes: colorComponents,
                        bytesPerRow: Int(size.width) * 4)
        
        return texture
    }
    
    func fillMipLevel(_ commandBuffer: MTLCommandBuffer, _ texture: MTLTexture, _ level: Int) {
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            fatalError("Failed to create blit command encoder")
        }

        blitEncoder.copy(from: mipColorTextures[level], sourceSlice: 0, sourceLevel: 0, to: texture, destinationSlice: 0, destinationLevel: level, sliceCount: 1, levelCount: 1)

        blitEncoder.endEncoding()
    }
    
    func copyTextureToMipLevel(_ commandBuffer: MTLCommandBuffer, _ textureDst: MTLTexture,  _ textureSrc: MTLTexture, _ level: Int) {
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else {
            fatalError("Failed to create blit command encoder")
        }

        blitEncoder.copy(from: textureSrc, sourceSlice: 0, sourceLevel: level, to: textureDst, destinationSlice: 0, destinationLevel: level, sliceCount: 1, levelCount: 1)

        blitEncoder.endEncoding()
    }
    
    func quaternionToEulerAngles(_ quaternion: simd_quatf) -> simd_float3 {
        // Extract the individual components of the quaternion
        let x = quaternion.imag.x
        let y = quaternion.imag.y
        let z = quaternion.imag.z
        let w = quaternion.real
        
        // Compute the Euler angles
        let sinPitch = 2.0 * (w * x + y * z)
        let cosPitch = 1.0 - 2.0 * (x * x + y * y)
        let pitch: Float = atan2(sinPitch, cosPitch)
        
        let sinYaw = 2.0 * (w * y - z * x)
        var yaw: Float = 0.0
        if (abs(sinYaw) < 1.0) {
            yaw = Float(asin(sinYaw))
        } else {
            yaw = Float(copysign(Float.pi / 2, sinYaw))
        }
        
        let sinRoll = 2.0 * (w * z + x * y)
        let cosRoll = 1.0 - 2.0 * (y * y + z * z)
        let roll: Float = atan2(sinRoll, cosRoll)
        
        return simd_float3(x: pitch, y: yaw, z: roll)
    }
    
    // https://www.gamedev.net/forums/topic/473037-removing-the-roll-from-a-quaternion/ kinda
    func RemovePitchAndRoll(_ rot: simd_quatf) -> simd_quatf
    {
        let angles = quaternionToEulerAngles(rot)

        return simd_quatf(angle: angles.y, axis: simd_float3(0,1,0)).normalized
    }
    
    var lastSubmit = 0.0
    var lastLastSubmit = 0.0
    func drawNextTexture(drawable: TextureResource.Drawable, simdDeviceAnchor: simd_float4x4, plane: ModelEntity, position: simd_float3, orientation: simd_quatf, scale: simd_float3) {
        autoreleasepool {
            guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                print("aaaaaaaaaa")
                return
            }
            
            for i in 0..<colorMipmapLevelStart {
                if onlyColorsNoTestImage {
                    fillMipLevel(commandBuffer, drawable.texture, i)
                }
                else {
                    copyTextureToMipLevel(commandBuffer, drawable.texture, testImageTexture, i)
                }
                if !enableDrawableMipmaps {
                    break
                }
            }
            if enableDrawableMipmaps {
                for i in colorMipmapLevelStart..<drawable.texture.mipmapLevelCount {
                    if colorMipLevels {
                        fillMipLevel(commandBuffer, drawable.texture, i)
                    }
                    else {
                        copyTextureToMipLevel(commandBuffer, drawable.texture, testImageTexture, i)
                    }
                }
            }
            
            if imageDoesntFollowHeadAtAll {
                plane.position = simd_float3(0.0, 1.0, -1.0)
                plane.orientation = simd_quatf(angle: 1.5708, axis: simd_float3(1,0,0))
                plane.scale = scale
            }
            else {
                plane.position = position
                plane.orientation = orientation
                plane.scale = scale
            }
            
            
            
            //commandBuffer.present(drawable)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted() // this is a load-bearing wait
            
        }
    }
}
