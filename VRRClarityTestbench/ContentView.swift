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

    @EnvironmentObject var gStore: GlobalSettingsStore

    @State private var showImmersiveSpace = false
    @State private var immersiveSpaceIsShown = false

    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace
    @Environment(\.dismissWindow) private var dismissWindow
    
    let imageFilteringMethodsAllowed = ["nearest", "bilinear", "bicubic"]
    
    let numberFormatter: NumberFormatter = {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter
    }()

    var body: some View {
        VStack {
            Text("VRR Clarity Testbench")
                
            Toggle(isOn: $gStore.settings.headlockTestImage)
            {
                Text("Headlock Test Image")
                Text("Headlock the test image including pitch/roll. If false, only headlock yaw.")
                    .font(.system(size: 10))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .toggleStyle(.switch)
            
            Toggle(isOn: $gStore.settings.imageDoesntFollowHeadAtAll)
            {
                Text("Image Doesn't Follow Head At All")
                Text("Just place the image in the world, no headlocking")
                    .font(.system(size: 10))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .toggleStyle(.switch)
            
            
            Toggle(isOn: $gStore.settings.colorMipLevels)
            {
                Text("Color Mipmap Levels")
                Text("Display different mipmap levels (below 1x) with a yellow -> orange -> red gradient")
                    .font(.system(size: 10))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .toggleStyle(.switch)
            
            Toggle(isOn: $gStore.settings.onlyColorsNoTestImage)
            {
                Text("Only Colors, No Test Image")
                Text("Level 0/1x shows as solid green, not a test image")
                    .font(.system(size: 10))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .toggleStyle(.switch)
            
            Toggle(isOn: $gStore.settings.enableDrawableMipmaps)
            {
                Text("Enable Drawable Mipmaps")
                Text("Test the test texture without mipmaps on, if false")
                    .font(.system(size: 10))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .toggleStyle(.switch)
            
            HStack {
                Text("Image Filtering Method")
                Picker("Image Filtering Method", selection: $gStore.settings.imageFilteringMethod) {
                    ForEach(imageFilteringMethodsAllowed, id: \.self) {
                        Text($0)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity, alignment: .trailing)
            }
            
            Text("Virtual Screen Depth (in)").frame(maxWidth: .infinity, alignment: .leading)
            HStack {
                Slider(value: $gStore.settings.virtualScreenDepth,
                      in: 0...200,
                      step: 0.5) {
                   Text("Virtual Screen Depth (in)")
               }
               
               TextField("Virtual Screen Depth (in)", value: $gStore.settings.virtualScreenDepth, formatter: numberFormatter)
               .textFieldStyle(RoundedBorderTextFieldStyle())
               .frame(width: 100)
            }
            
            Text("Virtual Screen Diagonal (in)").frame(maxWidth: .infinity, alignment: .leading)
            HStack {
                Slider(value: $gStore.settings.virtualScreenDiagonal,
                      in: 0...200,
                      step: 0.5) {
                   Text("Virtual Screen Diagonal (in)")
               }
               
               TextField("Virtual Screen Diagonal (in)", value: $gStore.settings.virtualScreenDiagonal, formatter: numberFormatter)
               .textFieldStyle(RoundedBorderTextFieldStyle())
               .frame(width: 100)
            }
            
            Text("Color Mipmap Level Start").frame(maxWidth: .infinity, alignment: .leading)
            HStack {
                Slider(value: $gStore.settings.colorMipmapLevelStart,
                      in: 1...10,
                      step: 1) {
                   Text("Color Mipmap Level Start")
               }
               
               TextField("Virtual Screen Diagonal (in)", value: $gStore.settings.colorMipmapLevelStart, formatter: numberFormatter)
                   .textFieldStyle(RoundedBorderTextFieldStyle())
                   .frame(width: 100)
            }

            Toggle("Show Virtual Screen", isOn: $showImmersiveSpace)
                .font(.title)
                .frame(width: 360)
                .padding(24)
                .glassBackgroundEffect()
            
            Button("Reset Settings") {
                gStore.reset()
                VRRClarityTestbenchApp.saveSettings()
            }
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
                    
                    //dismissWindow(id: "Entry")
                    
                } else if immersiveSpaceIsShown {
                    await dismissImmersiveSpace()
                    immersiveSpaceIsShown = false
                }
            }
        }
        .frame(minWidth: 650, maxWidth: 650)
        .glassBackgroundEffect()
    }
}

#Preview(windowStyle: .automatic) {
    ContentView()
}
