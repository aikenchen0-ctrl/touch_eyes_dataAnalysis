import UIKit

enum AispectHapticOption: CaseIterable {
    case impactLight
    case impactMedium
    case impactHeavy
    case impactSoft
    case impactRigid
    case notificationSuccess
    case notificationWarning
    case notificationError
    case selection

    var displayName: String {
        switch self {
        case .impactLight:
            return "impact light"
        case .impactMedium:
            return "impact medium"
        case .impactHeavy:
            return "impact heavy"
        case .impactSoft:
            return "impact soft"
        case .impactRigid:
            return "impact rigid"
        case .notificationSuccess:
            return "notification success"
        case .notificationWarning:
            return "notification warning"
        case .notificationError:
            return "notification error"
        case .selection:
            return "selection"
        }
    }

    var impactStyle: UIImpactFeedbackGenerator.FeedbackStyle? {
        switch self {
        case .impactLight:
            return .light
        case .impactMedium:
            return .medium
        case .impactHeavy:
            return .heavy
        case .impactSoft:
            return .soft
        case .impactRigid:
            return .rigid
        default:
            return nil
        }
    }

    var notificationType: UINotificationFeedbackGenerator.FeedbackType? {
        switch self {
        case .notificationSuccess:
            return .success
        case .notificationWarning:
            return .warning
        case .notificationError:
            return .error
        default:
            return nil
        }
    }

    var isSelection: Bool {
        self == .selection
    }
}
