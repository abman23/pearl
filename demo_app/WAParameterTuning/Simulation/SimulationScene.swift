/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The SpriteKit scene running the simulation.
*/

import Foundation
import SpriteKit
import SwiftUI

class SimulationScene: SKScene, SKPhysicsContactDelegate {
    private var satellite: SKSpriteNode?
    private var orbitRadius: CGFloat?

    private var mode: SimulationEngine.Mode?
    public var localEvents: AsyncStream<LocalEvent>?
    private var localEventsContinutation: AsyncStream<LocalEvent>.Continuation?

    override func didMove(to view: SKView) {
        physicsWorld.contactDelegate = self
    }

    func setup(as mode: SimulationEngine.Mode) {
        self.mode = mode
        (stream: localEvents, continuation: localEventsContinutation) = AsyncStream<LocalEvent>.makeStream()

        removeAllChildren()

        setBackground()

        let planet = makePlanet(at: .init(x: frame.midX, y: frame.midY), scalingFactor: 0.45)
        addChild(planet)

        satellite = makeSatellite(scalingFactor: 0.15)
        guard let satellite else { return }
        if mode == .host {
            orbitRadius = (planet.size.width / 2)

            guard let orbitRadius else { return }
            satellite.position = CGPoint(x: planet.position.x + orbitRadius, y: 0)
            enableSatellite()
            enableSatelliteOrbit(orbitRadius)
        }
    }

    func setBackground() {
        let background = SKSpriteNode(imageNamed: "OuterSpace")
        background.zPosition = 0
        background.position = CGPoint(x: self.size.width / 2, y: self.size.height / 2)
        background.size = CGSize(width: self.size.width, height: self.size.height)

        addChild(background)
    }

    func makePlanet(at position: CGPoint, scalingFactor: CGFloat) -> SKSpriteNode {
        let planet = SKSpriteNode(imageNamed: "Planet")
        planet.setScale((scalingFactor * frame.width) / planet.size.width)
        planet.position = position

        return planet
    }

    func makeSatellite(scalingFactor: CGFloat) -> SKSpriteNode? {
        let satellite = SKSpriteNode(imageNamed: "Satellite")
        satellite.setScale((scalingFactor * frame.width) / satellite.size.width)

        return satellite
    }

    func enableSatellite() {
        guard let satellite else { return }
        disableSatellite()
        addChild(satellite)
    }

    func disableSatellite() {
        guard let satellite else { return }
        removeChildren(in: [satellite])
    }

    private func enableSatelliteOrbit(_ orbitRadius: CGFloat?) {
        guard let satellite, let orbitRadius else { return }

        let orbit = UIBezierPath.circlePath(radius: orbitRadius, at: .init(x: frame.midX, y: frame.midY))
        let orbitPath = SKAction.follow(orbit, asOffset: false, orientToPath: false, speed: 0.3 * frame.width)
        satellite.run(SKAction.repeatForever(orbitPath))
    }

    private func disableSatelliteOrbit() {
        guard let satellite else { return }
        satellite.removeAllActions()
    }

    func moveSatellite(to position: CGPoint, using dimensions: CGSize) {
        let transformRatioX: Double = frame.width / dimensions.width
        let transformRatioY: Double = frame.height / dimensions.height

        satellite?.position = CGPoint(x: transformRatioX * position.x, y: transformRatioY * position.y)
    }

    override func update(_ currentTime: TimeInterval) {
        if mode == .host, let position = satellite?.position {
            localEventsContinutation?.yield(.satelliteMovedTo(position))
        }
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        if mode == .host {
            disableSatelliteOrbit()
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        if mode == .host, let radius = orbitRadius {
            guard let touch = touches.first else { return }
            let touchLocation = touch.location(in: self)

            // Calculate the angle from the center to the touch
            let angle = atan2(touchLocation.y - frame.midY, touchLocation.x - frame.midX)

            // Calculate the new position on the edge of the orbit
            let newX = frame.midX + radius * cos(angle)
            let newY = frame.midY + radius * sin(angle)

            // Set the position to the edge of the orbit
            satellite?.position = CGPoint(x: newX, y: newY)
        }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        if mode == .host {
            enableSatelliteOrbit(orbitRadius)
        }
    }
}

extension UIBezierPath {
    static func circlePath(radius: CGFloat, at center: CGPoint) -> CGPath {
        return UIBezierPath(arcCenter: center, radius: radius, startAngle: 0, endAngle: 2 * .pi, clockwise: true).cgPath
    }
}
