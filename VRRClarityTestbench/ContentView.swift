//
//  ContentView.swift
//  VRRClarityTestbench
//
//  Created by Max Thomas on 4/24/24.
//

import SwiftUI
import RealityKit
import RealityKitContent

struct ContentView: View {

    @State private var showImmersiveSpace = false
    @State private var immersiveSpaceIsShown = false

    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace
    @Environment(\.dismissWindow) private var dismissWindow

    var body: some View {
        VStack {
            Model3D(named: "Scene", bundle: realityKitContentBundle)
                .padding(.bottom, 50)

            Text("Hello, world!")

            Toggle("Show ImmersiveSpace", isOn: $showImmersiveSpace)
                .font(.title)
                .frame(width: 360)
                .padding(24)
                .glassBackgroundEffect()
        }
        .padding()
        .onChange(of: showImmersiveSpace) { _, newValue in
            Task {
                if newValue && !immersiveSpaceIsShown {
                    
                    if !DummyMetalRenderer.haveRenderInfo {
                        var dummySpaceIsOpened = false
                        while !dummySpaceIsOpened {
                            switch await openImmersiveSpace(id: "DummyImmersiveSpace") {
                            case .opened:
                                dummySpaceIsOpened = true
                            case .error, .userCancelled:
                                fallthrough
                            @unknown default:
                                dummySpaceIsOpened = false
                            }
                        }
                        
                        while dummySpaceIsOpened && !DummyMetalRenderer.haveRenderInfo {
                            try! await Task.sleep(nanoseconds: 1_000_000)
                        }
                        
                        await dismissImmersiveSpace()
                        try! await Task.sleep(nanoseconds: 1_000_000_000)
                    }
                    
                    if !DummyMetalRenderer.haveRenderInfo {
                        print("MISSING VIEW INFO!!")
                    }
                    
                    print("Open real immersive space")
                    
                    switch await openImmersiveSpace(id: "ImmersiveSpace") {
                    case .opened:
                        immersiveSpaceIsShown = true
                    case .error, .userCancelled:
                        fallthrough
                    @unknown default:
                        immersiveSpaceIsShown = false
                        showImmersiveSpace = false
                    }
                    
                    dismissWindow(id: "Entry")
                    
                } else if immersiveSpaceIsShown {
                    await dismissImmersiveSpace()
                    immersiveSpaceIsShown = false
                }
            }
        }
    }
}

#Preview(windowStyle: .automatic) {
    ContentView()
}
