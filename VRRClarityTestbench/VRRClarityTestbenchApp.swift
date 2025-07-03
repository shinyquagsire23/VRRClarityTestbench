//
//  VRRClarityTestbenchApp.swift
//  VRRClarityTestbench
//
//  Created by Max Thomas on 4/24/24.
//

import SwiftUI
import CompositorServices

struct ContentStageConfiguration: CompositorLayerConfiguration {
    func makeConfiguration(capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration) {
        configuration.depthFormat = .depth32Float
        configuration.colorFormat = .bgra8Unorm_srgb
    
        let foveationEnabled = capabilities.supportsFoveation
        configuration.isFoveationEnabled = foveationEnabled
        
        let options: LayerRenderer.Capabilities.SupportedLayoutsOptions = foveationEnabled ? [.foveationEnabled] : []
        let supportedLayouts = capabilities.supportedLayouts(options: options)
        
        configuration.layout = supportedLayouts.contains(.layered) ? .layered : .dedicated
        
        configuration.colorFormat = .rgba16Float
    }
}

@main
struct VRRClarityTestbenchApp: App {
    @Environment(\.scenePhase) private var scenePhase
    static var gStore = GlobalSettingsStore()
    
    static func saveSettings() {
        do {
            try VRRClarityTestbenchApp.gStore.save(settings: VRRClarityTestbenchApp.gStore.settings)
        } catch {
            fatalError(error.localizedDescription)
        }
    }
    
    static func loadSettings() {
        do {
            try VRRClarityTestbenchApp.gStore.load()
        } catch {
            fatalError(error.localizedDescription)
        }
    }
    
    var body: some Scene {
        WindowGroup(id: "Entry") {
            ContentView()
            .task {
                VRRClarityTestbenchApp.loadSettings()
            }
            .onChange(of: scenePhase) {
                switch scenePhase {
                case .background:
                    VRRClarityTestbenchApp.saveSettings()
                    break
                case .inactive:
                    VRRClarityTestbenchApp.saveSettings()
                    break
                case .active:
                    VRRClarityTestbenchApp.loadSettings()
                    break
                @unknown default:
                    break
                }
            }
            .environmentObject(VRRClarityTestbenchApp.gStore)
            .fixedSize()
        }
        .windowStyle(.plain)
        .windowResizability(.contentSize)
        
        ImmersiveSpace(id: "DummyImmersiveSpace") {
            CompositorLayer(configuration: ContentStageConfiguration()) { layerRenderer in
                let renderer = DummyMetalRenderer(layerRenderer)
                renderer.startRenderLoop()
            }
        }.immersionStyle(selection: .constant(.full), in: .full)

        ImmersiveSpace(id: "ImmersiveSpace") {
            ImmersiveView()
        }.immersionStyle(selection: .constant(.mixed), in: .mixed)
    }
}
