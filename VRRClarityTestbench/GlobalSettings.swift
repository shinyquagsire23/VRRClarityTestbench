//
//  GlobalSettings.swift
//  VRRClarityTestbench
//
//  Created by Max Thomas on 7/3/25.
//


//
//  GlobalSettings.swift
//
// Client-side settings and defaults
//

import Foundation
import SwiftUI

struct GlobalSettings: Codable {
    //
    // Test suite parameters
    // -----------------------------------------------------------------
    // Headlock the test image including pitch/roll. If false, only headlock yaw.
    var headlockTestImage = false

    // Just place the image in the world, no headlocking
    var imageDoesntFollowHeadAtAll = false

    // Display different mipmap levels (below 1x) with a yellow -> orange -> red gradient
    var colorMipLevels = true

    // Level 0/1x shows as solid green, not a test image
    var onlyColorsNoTestImage = false

    // Test the test texture without mipmaps on, if false
    var enableDrawableMipmaps = true

    // How to filter the image when it is drawn by RealityKit
    var imageFilteringMethod = "bicubic"

    // Virtual screen size/depth
    var virtualScreenDepth: Float = 30.0 // 30in away
    var virtualScreenDiagonal: Float = 28.0

    var colorMipmapLevelStart: Float = 1 // set to 2 for 4k textures, or to view the texture fully.
    
    init() {}
    
    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        
        self.headlockTestImage = try container.decodeIfPresent(Bool.self, forKey: .headlockTestImage) ?? self.headlockTestImage
        self.imageDoesntFollowHeadAtAll = try container.decodeIfPresent(Bool.self, forKey: .imageDoesntFollowHeadAtAll) ?? self.imageDoesntFollowHeadAtAll
        self.colorMipLevels = try container.decodeIfPresent(Bool.self, forKey: .colorMipLevels) ?? self.colorMipLevels
        self.onlyColorsNoTestImage = try container.decodeIfPresent(Bool.self, forKey: .onlyColorsNoTestImage) ?? self.onlyColorsNoTestImage
        self.enableDrawableMipmaps = try container.decodeIfPresent(Bool.self, forKey: .enableDrawableMipmaps) ?? self.enableDrawableMipmaps
        
        self.imageFilteringMethod = try container.decodeIfPresent(String.self, forKey: .imageFilteringMethod) ?? self.imageFilteringMethod
        
        self.virtualScreenDepth = try container.decodeIfPresent(Float.self, forKey: .virtualScreenDepth) ?? self.virtualScreenDepth
        self.virtualScreenDiagonal = try container.decodeIfPresent(Float.self, forKey: .virtualScreenDiagonal) ?? self.virtualScreenDiagonal
        
        self.colorMipmapLevelStart = try container.decodeIfPresent(Float.self, forKey: .colorMipmapLevelStart) ?? self.colorMipmapLevelStart
    }
}

extension GlobalSettingsStore {
    static let sampleData: GlobalSettingsStore =
    GlobalSettingsStore()
}

class GlobalSettingsStore: ObservableObject {
    @Published var settings: GlobalSettings = GlobalSettings()
    
    private static func fileURL() throws -> URL {
        try FileManager.default.url(for: .documentDirectory,
                                    in: .userDomainMask,
                                    appropriateFor: nil,
                                    create: true)
        .appendingPathComponent("globalsettings.data")
    }
    
    func load() throws {
        let fileURL = try Self.fileURL()
        guard let data = try? Data(contentsOf: fileURL) else {
            return self.settings = GlobalSettings()
        }
        let globalSettings = try JSONDecoder().decode(GlobalSettings.self, from: data)
        self.settings = globalSettings
    }
    
    func save(settings: GlobalSettings) throws {
        let data = try JSONEncoder().encode(settings)
        let outfile = try Self.fileURL()
        try data.write(to: outfile)
    }
    
    func reset() {
        self.settings = GlobalSettings()
    }
}
