import UIKit

protocol LiveActivityProviding {
    func startLiveActivity()
    func stopLiveActivity()
}

protocol PictureInPictureProviding {
    func startPictureInPicture()
    func stopPictureInPicture()
}

protocol FloatingHomeProviding {
    func makeFloatingHomeView() -> UIView
}

protocol TodoPlannerProviding {
    func proposePlan(for intent: String) -> [String]
}

protocol KnowledgeBaseProviding {
    func index(content: String)
    func search(query: String) -> [String]
}
